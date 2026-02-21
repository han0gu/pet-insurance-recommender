from __future__ import annotations

import os
import re
from contextlib import nullcontext
from dataclasses import dataclass
from typing import Any, Dict, List, Literal, Tuple

from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from langchain_core.documents import Document
from langchain_upstage.document_parse import OutputFormat
from pydantic import BaseModel, Field

try:
    from langsmith.run_helpers import tracing_context
except Exception:  # pragma: no cover - optional dependency fallback
    tracing_context = None

from app.agents.document_parser.constants import TAG_TYPE
from app.agents.document_parser.nodes.path_utils import build_tagged_chunks_dir
from app.agents.document_parser.nodes.tagger.chunk_file import create_chunk_file
from app.agents.document_parser.nodes.tagger.chunk_loader import load_chunk_files
from app.agents.document_parser.nodes.tagger.tag_summary import summarize_counts


load_dotenv()

LLM_USE_POLICY = Literal["always", "unknown_or_low_conf", "never"]

DEFAULT_EMBEDDING_MODEL = "solar-embedding-1-large"
DEFAULT_USE_LLM_WHEN: LLM_USE_POLICY = "never"
DEFAULT_LLM_CONF_THRESHOLD = 0.50

_STRUCTURED_LLM_CACHE: Dict[Tuple[str], Any] = {}
_TAG_RESULT_CACHE: Dict[Tuple[str, str, str, float], Dict[str, Any]] = {}
_TAG_RESULT_CACHE_MAX = int(os.getenv("TAGGING_RESULT_CACHE_MAX", "5000"))
_UPSTAGE_BASE_URL = "https://api.upstage.ai/v1/solar"
_SOLAR_MODEL = "solar-pro2"

RISK_DOMAINS = [
    "head",
    "dental",
    "skin",
    "joint",
    "urinary",
    "eye",
    "digestive",
    "other",
]
TERM_TYPES = ["basic", "special", "other"]

RISK_DOMAIN_RULES: List[Tuple[str, str]] = [
    (r"(뇌|두부|머리|경련|신경)", "head"),
    (r"(치아|치주|스케일링|구강)", "dental"),
    (r"(피부|습진|알레르기|가려움)", "skin"),
    (r"(관절|슬개골|탈구|고관절|십자인대)", "joint"),
    (r"(비뇨|방광|요로|신장|결석)", "urinary"),
    (r"(눈|각막|백내장|망막)", "eye"),
    (r"(위|장|소화|구토|설사)", "digestive"),
]


@dataclass(frozen=True)
class TaggerProfile:
    tag_type: TAG_TYPE
    clause_types: List[str]
    clause_type_rules: List[Tuple[str, str]]
    user_prompt_lines: List[str]


PROFILES: Dict[str, TaggerProfile] = {
    "normal": TaggerProfile(
        tag_type="normal",
        clause_types=[
            "coverage",
            "exclusion",
            "waiting",
            "deductible",
            "limit",
            "claim",
            "definition",
            "renewal",
            "other",
        ],
        clause_type_rules=[
            (r"(면책|보상하지\s*않|지급하지\s*않|제외)", "exclusion"),
            (r"(대기기간|면책기간|경과\s*\d+\s*일)", "waiting"),
            (r"(자기부담|공제금|본인부담)", "deductible"),
            (r"(한도|지급한도|연간\s*한도|1회\s*한도|최대\s*지급)", "limit"),
            (r"(보험금\s*청구|청구\s*서류|접수|지급\s*절차)", "claim"),
            (r"(정의|용어의\s*정의)", "definition"),
            (r"(갱신|재가입|갱신형)", "renewal"),
            (r"(보장|지급\s*사유|보험금\s*지급)", "coverage"),
        ],
        user_prompt_lines=[
            "- exclusion: 보상하지 않음/지급하지 않음/면책/제외",
            "- waiting: 대기기간/면책기간/경과 N일",
            "- deductible: 자기부담/본인부담/공제금",
            "- limit: 한도/최대/연간한도/1회한도",
            "- coverage: 지급 사유/보장/보험금 지급",
            "- claim: 청구 절차/서류/접수",
            "- definition: 용어 정의",
            "- renewal: 갱신/재가입",
        ],
    ),
    "simple": TaggerProfile(
        tag_type="simple",
        clause_types=["coverage", "exclusion", "other"],
        clause_type_rules=[
            (r"(면책|보상하지\s*않|지급하지\s*않|제외|부지급)", "exclusion"),
            (r"(보장|지급\s*사유|보험금\s*지급|보상)", "coverage"),
        ],
        user_prompt_lines=[
            "- exclusion: 보상하지 않음/지급하지 않음/면책/제외/부지급",
            "- coverage: 보장 범위/지급 사유/보험금 지급/보상",
            "- other: 위 두 가지에 명확히 해당하지 않는 내용",
        ],
    ),
}


class ChunkTagOutput(BaseModel):
    clause_type: str = Field(...)
    risk_domains: List[str] = Field(min_length=1)
    confidence: float = Field(ge=0, le=1)
    notes: str | None = Field(default=None, max_length=200)


def _get_tagging_langsmith_project_name() -> str | None:
    project_name = os.getenv("LANGSMITH_TAGGING_PROJECT")
    if project_name:
        return project_name
    return None


def _get_profile(tag_type: TAG_TYPE) -> TaggerProfile:
    profile = PROFILES.get(tag_type)
    if profile is None:
        raise ValueError(f"Unsupported tag_type: {tag_type}")
    return profile


def _rule_tag(text: str, profile: TaggerProfile) -> Dict[str, Any]:
    clause_type = "other"
    for pattern, candidate in profile.clause_type_rules:
        if re.search(pattern, text):
            clause_type = candidate
            break

    domains: List[str] = []
    for pattern, domain in RISK_DOMAIN_RULES:
        if re.search(pattern, text):
            domains.append(domain)
    domains = sorted(set(domains)) or ["other"]

    confidence = 0.55 if clause_type != "other" else 0.25
    return {
        "clause_type": clause_type,
        "risk_domains": domains,
        "confidence": confidence,
        "method": "rule",
        "notes": None,
    }


def _llm_tag_solar_pro2(
    text: str,
    *,
    profile: TaggerProfile,
    api_key: str,
) -> Dict[str, Any]:
    os.environ["UPSTAGE_API_KEY"] = api_key
    os.environ.setdefault("UPSTAGE_BASE_URL", _UPSTAGE_BASE_URL)

    system = (
        "너는 보험 약관 문서 청크를 분류(tagging)하는 분류기야.\n"
        "아래 라벨 셋으로만 분류해.\n"
        f"- clause_type: {', '.join(profile.clause_types)}\n"
        f"- risk_domains: {', '.join(RISK_DOMAINS)}\n"
        "반드시 JSON schema에 맞는 JSON만 출력해."
    )
    user = (
        "다음 텍스트 청크를 라벨링해줘.\n"
        "분류 기준:\n"
        f"{chr(10).join(profile.user_prompt_lines)}\n"
        "risk_domains는 텍스트에 명시된 신체/질환 영역을 추정해.\n\n"
        f"TEXT:\n{text}"
    )

    cache_key = (_SOLAR_MODEL,)
    structured_llm = _STRUCTURED_LLM_CACHE.get(cache_key)
    if structured_llm is None:
        llm = init_chat_model(model=_SOLAR_MODEL, temperature=0.0)
        structured_llm = llm.with_structured_output(ChunkTagOutput)
        _STRUCTURED_LLM_CACHE[cache_key] = structured_llm

    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]
    tagging_project = _get_tagging_langsmith_project_name()
    tracing_cm = (
        tracing_context(project_name=tagging_project)
        if tracing_context and tagging_project
        else nullcontext()
    )
    with tracing_cm:
        llm_response: ChunkTagOutput = structured_llm.invoke(messages)
    data = llm_response.model_dump()
    data["method"] = "llm"
    return data


def _validate_and_override(
    text: str, tag: Dict[str, Any], profile: TaggerProfile
) -> Dict[str, Any]:
    if re.search(r"(보상하지\s*않|지급하지\s*않|면책|제외|부지급)", text):
        tag["clause_type"] = "exclusion"
        tag["confidence"] = max(tag.get("confidence", 0.0), 0.85)
        tag["notes"] = (tag.get("notes") or "")[:150]

    if profile.tag_type == "normal":
        if re.search(r"(자기부담|본인부담|공제금)", text):
            tag["clause_type"] = "deductible"
            tag["confidence"] = max(tag.get("confidence", 0.0), 0.85)
            tag["notes"] = (tag.get("notes") or "")[:150]
        if re.search(r"(지급한도|연간\s*한도|1회\s*한도|최대\s*지급|한도)", text):
            if tag.get("clause_type") != "deductible":
                tag["clause_type"] = "limit"
                tag["confidence"] = max(tag.get("confidence", 0.0), 0.80)
                tag["notes"] = (tag.get("notes") or "")[:150]
    else:
        if re.search(r"(보장|지급\s*사유|보험금\s*지급|보상)", text):
            if tag.get("clause_type") != "exclusion":
                tag["clause_type"] = "coverage"
                tag["confidence"] = max(tag.get("confidence", 0.0), 0.80)
                tag["notes"] = (tag.get("notes") or "")[:150]

    if not tag.get("risk_domains"):
        tag["risk_domains"] = ["other"]
    if tag["clause_type"] not in profile.clause_types:
        tag["clause_type"] = "other"
    tag["risk_domains"] = [
        domain if domain in RISK_DOMAINS else "other" for domain in tag["risk_domains"]
    ]
    if not tag["risk_domains"]:
        tag["risk_domains"] = ["other"]
    return tag


def _tag_chunk(
    text: str,
    *,
    profile: TaggerProfile,
    upstage_api_key: str,
    use_llm_when: LLM_USE_POLICY,
    llm_conf_threshold: float,
) -> Dict[str, Any]:
    cache_key = (profile.tag_type, text, use_llm_when, llm_conf_threshold)
    cached = _TAG_RESULT_CACHE.get(cache_key)
    if cached is not None:
        return dict(cached)

    base = _rule_tag(text, profile)
    should_llm = False
    if use_llm_when == "always":
        should_llm = True
    elif use_llm_when == "unknown_or_low_conf":
        should_llm = (
            base["clause_type"] == "other" or base["confidence"] < llm_conf_threshold
        )

    if should_llm:
        try:
            merged = _llm_tag_solar_pro2(
                text,
                profile=profile,
                api_key=upstage_api_key,
            )
        except Exception as exc:
            merged = base
            merged["notes"] = f"LLM failed: {type(exc).__name__}"
    else:
        merged = base

    merged = _validate_and_override(text, merged, profile)

    if len(_TAG_RESULT_CACHE) >= _TAG_RESULT_CACHE_MAX:
        _TAG_RESULT_CACHE.pop(next(iter(_TAG_RESULT_CACHE)))
    _TAG_RESULT_CACHE[cache_key] = dict(merged)
    return merged


def tag_chunks(
    *,
    chunks: List[Document],
    output_format: OutputFormat,
    tag_type: TAG_TYPE,
    embedding_model: str = DEFAULT_EMBEDDING_MODEL,
    use_llm_when: LLM_USE_POLICY = DEFAULT_USE_LLM_WHEN,
    llm_conf_threshold: float = DEFAULT_LLM_CONF_THRESHOLD,
    overwrite: bool = False,
) -> List[Document]:
    if not chunks:
        raise ValueError("chunks is empty")

    upstage_api_key = os.getenv("UPSTAGE_API_KEY", "")
    if use_llm_when != "never" and not upstage_api_key:
        raise ValueError("UPSTAGE_API_KEY is not set. Please check your .env file.")

    file_name = chunks[0].metadata["doc"]["file_name"]

    if not overwrite:
        existing_chunks = load_chunk_files(
            file_name=file_name,
            output_format=output_format,
            tag_type=tag_type,
        )
        if existing_chunks:
            print(
                f"♻️ tagger: found existing tagged chunks. skip tagging and load existing tagged chunks. {len(existing_chunks)} (tag_type={tag_type})"
            )
            return existing_chunks

    profile = _get_profile(tag_type)
    target_dir = build_tagged_chunks_dir(
        file_name=file_name,
        output_format=output_format,
        tag_type=tag_type,
    )
    tagged_chunks: List[Document] = []
    llm_used_count = 0
    for idx, chunk in enumerate(chunks):
        chunk_text = chunk.page_content
        tag = _tag_chunk(
            chunk_text,
            profile=profile,
            upstage_api_key=upstage_api_key,
            use_llm_when=use_llm_when,
            llm_conf_threshold=llm_conf_threshold,
        )

        metadata = {
            **chunk.metadata,
            "clause": {
                "clause_type": tag["clause_type"],
                "risk_domains": tag["risk_domains"],
            },
            "indexing": {
                "chunk_id": f"chunk_{idx:06d}",
                "chunk_char_len": len(chunk_text),
                "embedding_model": embedding_model,
                "tag_type": tag_type,
                "tag_method": tag["method"],
                "tag_confidence": tag["confidence"],
            },
        }
        tagged_chunk = Document(page_content=chunk_text, metadata=metadata)
        tagged_chunks.append(tagged_chunk)

        output_file_name = f"chunk_{idx:06d}.py"
        create_chunk_file(
            chunk=tagged_chunk,
            target_dir=target_dir,
            output_file_name=output_file_name,
        )

        if tag["method"] == "llm":
            llm_used_count += 1
        if (idx + 1) % 25 == 0 or idx == len(chunks) - 1:
            print(
                f"🚀[tagging] processed {idx + 1}/{len(chunks)} chunks "
                f"(llm_used={llm_used_count})"
            )

    summarize_counts(
        tagged_chunks=tagged_chunks,
        tag_type=tag_type,
        output_format=output_format,
        clause_types=profile.clause_types,
        term_types=TERM_TYPES,
    )
    return tagged_chunks
