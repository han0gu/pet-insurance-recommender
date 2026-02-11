from __future__ import annotations

import json
import os
import re
from typing import Any, Dict, List, Tuple

from dotenv import load_dotenv
from openai import OpenAI

from langchain_core.documents import Document

# `.env` 파일의 환경변수를 현재 프로세스에 로드합니다.
# - 로컬 개발 시 `UPSTAGE_API_KEY`를 코드에 하드코딩하지 않고 사용하기 위함입니다.
load_dotenv()


# =========================
# 0) PoC 라벨 정의 (Label Set)
# =========================
# 본 파일은 각 chunk를 아래 2개 축으로 분류합니다.
# 1) clause_type: 약관 조항의 성격 (보장/면책/한도 등)
# 2) risk_domains: 어떤 신체/질환 영역인지
# 마지막 "other"는 규칙/LLM으로 분류가 애매할 때의 안전한 기본값입니다.
CLAUSE_TYPES = [
    "coverage",
    "exclusion",
    "waiting",
    "deductible",
    "limit",
    "claim",
    "definition",
    "renewal",
    "other",
]
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


# =========================
# 1) 규칙 기반(Regex) 1차 태깅
# =========================
# 텍스트에 특정 키워드가 있으면 빠르게 초기 라벨을 부여합니다.
# 순서가 중요합니다. 먼저 매칭된 규칙이 우선 적용됩니다.
# (예: 면책 키워드가 나오면 exclusion을 먼저 확정)
CLAUSE_TYPE_RULES: List[Tuple[str, str]] = [
    (r"(면책|보상하지\s*않|지급하지\s*않|제외)", "exclusion"),
    (r"(대기기간|면책기간|경과\s*\d+\s*일)", "waiting"),
    (r"(자기부담|공제금|본인부담)", "deductible"),
    (r"(한도|지급한도|연간\s*한도|1회\s*한도|최대\s*지급)", "limit"),
    (r"(보험금\s*청구|청구\s*서류|접수|지급\s*절차)", "claim"),
    (r"(정의|용어의\s*정의)", "definition"),
    (r"(갱신|재가입|갱신형)", "renewal"),
    (r"(보장|지급\s*사유|보험금\s*지급)", "coverage"),
]

RISK_DOMAIN_RULES: List[Tuple[str, str]] = [
    (r"(뇌|두부|머리|경련|신경)", "head"),
    (r"(치아|치주|스케일링|구강)", "dental"),
    (r"(피부|습진|알레르기|가려움)", "skin"),
    (r"(관절|슬개골|탈구|고관절|십자인대)", "joint"),
    (r"(비뇨|방광|요로|신장|결석)", "urinary"),
    (r"(눈|각막|백내장|망막)", "eye"),
    (r"(위|장|소화|구토|설사)", "digestive"),
]


def rule_tag(text: str) -> Dict[str, Any]:
    """
    규칙(정규식)만 사용해 1차 태깅을 수행합니다.

    Returns:
        {
            "clause_type": str,      # 조항 분류
            "risk_domains": list[str], # 위험영역(복수 가능)
            "confidence": float,     # 규칙 기반 신뢰도(보수적)
            "method": "rule",        # 어떤 방식으로 태깅했는지
            "notes": None            # 후처리/오류메모 영역
        }
    """
    clause_type = "other"
    for pat, ctype in CLAUSE_TYPE_RULES:
        # 첫 매칭만 채택해 과도한 덮어쓰기를 방지합니다.
        if re.search(pat, text):
            clause_type = ctype
            break

    domains: List[str] = []
    for pat, d in RISK_DOMAIN_RULES:
        if re.search(pat, text):
            domains.append(d)
    domains = sorted(set(domains)) or ["other"]

    # 규칙 기반은 빠르지만 문맥 이해가 제한적이므로
    # 신뢰도(confidence)는 보수적으로 시작합니다.
    confidence = 0.55 if clause_type != "other" else 0.25

    return {
        "clause_type": clause_type,
        "risk_domains": domains,
        "confidence": confidence,
        "method": "rule",
        "notes": None,
    }


# =========================
# 2) Solar Pro 2로 구조화 출력 태깅 (LLM Tagging)
# =========================
def llm_tag_solar_pro2(
    text: str,
    *,
    api_key: str,
    base_url: str = "https://api.upstage.ai/v1/solar",
    model: str = "solar-pro2",
    timeout_s: int = 30,
) -> Dict[str, Any]:
    """
    LLM(Upstage Solar Pro 2)으로 구조화 태깅을 수행합니다.

    핵심 포인트:
    - OpenAI 호환 Chat Completions API를 사용합니다.
    - response_format(json_schema)로 출력 형식을 강제합니다.
      -> 파싱 안정성이 높아지고 후처리 코드가 단순해집니다.

    Args:
        text: 태깅할 chunk 원문
        api_key: Upstage API Key
        base_url: Upstage OpenAI-compatible endpoint
        model: 사용할 모델명
        timeout_s: API 타임아웃(초)

    Returns:
        rule_tag와 동일 형태 + method="llm"
    """
    client = OpenAI(api_key=api_key, base_url=base_url, timeout=timeout_s)

    # 모델 출력 JSON 스키마 정의:
    # - enum으로 허용 라벨을 제한해 예상치 못한 값이 들어오지 않도록 합니다.
    # - strict=True로 스키마를 강하게 준수하도록 요청합니다.
    schema = {
        "name": "ChunkTag",
        "schema": {
            "type": "object",
            "additionalProperties": False,
            "properties": {
                "clause_type": {"type": "string", "enum": CLAUSE_TYPES},
                "risk_domains": {
                    "type": "array",
                    "items": {"type": "string", "enum": RISK_DOMAINS},
                    "minItems": 1,
                },
                "confidence": {"type": "number", "minimum": 0, "maximum": 1},
                "notes": {"type": ["string", "null"], "maxLength": 200},
            },
            "required": ["clause_type", "risk_domains", "confidence", "notes"],
        },
        "strict": True,
    }

    system = (
        "너는 보험 약관 문서 청크를 분류(tagging)하는 분류기야.\n"
        "아래 라벨 셋으로만 분류해.\n"
        f"- clause_type: {', '.join(CLAUSE_TYPES)}\n"
        f"- risk_domains: {', '.join(RISK_DOMAINS)}\n"
        "반드시 JSON schema에 맞는 JSON만 출력해."
    )

    user = (
        "다음 텍스트 청크를 라벨링해줘.\n"
        "분류 기준:\n"
        "- exclusion: 보상하지 않음/지급하지 않음/면책/제외\n"
        "- waiting: 대기기간/면책기간/경과 N일\n"
        "- deductible: 자기부담/본인부담/공제금\n"
        "- limit: 한도/최대/연간한도/1회한도\n"
        "- coverage: 지급 사유/보장/보험금 지급\n"
        "- claim: 청구 절차/서류/접수\n"
        "- definition: 용어 정의\n"
        "- renewal: 갱신/재가입\n"
        "risk_domains는 텍스트에 명시된 신체/질환 영역을 추정해.\n\n"
        f"TEXT:\n{text}"
    )

    resp = client.chat.completions.create(
        model=model,
        temperature=0,  # 분류 작업은 창의성보다 재현성이 중요합니다.
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        # Structured Output을 사용해 JSON 형태를 강제합니다.
        response_format={"type": "json_schema", "json_schema": schema},
    )

    # 모델 응답(JSON 문자열)을 dict로 변환합니다.
    content = resp.choices[0].message.content
    data = json.loads(content)

    # 다운스트림에서 태깅 출처를 알 수 있도록 method를 명시합니다.
    data["method"] = "llm"
    return data


# =========================
# 3) 검증/보정(Validation/Override)
# =========================
def validate_and_override(text: str, tag: Dict[str, Any]) -> Dict[str, Any]:
    """
    보험 도메인에서 강력한 신호는 룰로 override.
    LLM이 헷갈려도 여기서 안정화.
    """
    # 강한 패턴은 모델 결과보다 우선 적용해 분류 안정성을 높입니다.
    # (보험 약관에서 면책/자기부담/한도는 의미가 매우 크기 때문)

    # exclusion 강제 신호
    if re.search(r"(보상하지\s*않|지급하지\s*않|면책|제외)", text):
        tag["clause_type"] = "exclusion"
        tag["confidence"] = max(tag.get("confidence", 0.0), 0.85)
        tag["notes"] = (tag.get("notes") or "")[:150]

    # deductible 강제 신호
    if re.search(r"(자기부담|본인부담|공제금)", text):
        tag["clause_type"] = "deductible"
        tag["confidence"] = max(tag.get("confidence", 0.0), 0.85)
        tag["notes"] = (tag.get("notes") or "")[:150]

    # limit 강제 신호
    if re.search(r"(지급한도|연간\s*한도|1회\s*한도|최대\s*지급|한도)", text):
        # deductible이 더 우선일 때가 많아서, deductible이면 유지
        if tag.get("clause_type") != "deductible":
            tag["clause_type"] = "limit"
            tag["confidence"] = max(tag.get("confidence", 0.0), 0.80)
            tag["notes"] = (tag.get("notes") or "")[:150]

    # risk_domains 최소 1개 보장
    if not tag.get("risk_domains"):
        tag["risk_domains"] = ["other"]

    # enum 검증(안전장치):
    # 외부 입력/모델 이상 응답 등으로 허용되지 않은 값이 들어오면
    # 파이프라인이 깨지지 않도록 other로 정규화합니다.
    if tag["clause_type"] not in CLAUSE_TYPES:
        tag["clause_type"] = "other"
    tag["risk_domains"] = [
        d if d in RISK_DOMAINS else "other" for d in tag["risk_domains"]
    ]
    if not tag["risk_domains"]:
        tag["risk_domains"] = ["other"]

    return tag


# =========================
# 4) 최종: tag_chunk(text)
# =========================
def tag_chunk(
    text: str,
    *,
    upstage_api_key: str,
    use_llm_when: str = "unknown_or_low_conf",  # "always" | "unknown_or_low_conf" | "never"
    llm_conf_threshold: float = 0.70,
) -> Dict[str, Any]:
    """
    chunk 1개에 대한 최종 태깅 진입점입니다.

    동작 순서:
    1) 규칙 기반 1차 태깅(rule_tag)
    2) 설정(use_llm_when, llm_conf_threshold)에 따라 LLM 태깅 수행 여부 결정
    3) validate_and_override로 보정/검증

    실패 처리:
    - LLM 호출에 실패하면 규칙 기반 결과로 fallback합니다.
    """
    base = rule_tag(text)

    should_llm = False
    if use_llm_when == "always":
        should_llm = True
    elif use_llm_when == "never":
        should_llm = False
    else:
        # unknown 또는 확신도 낮으면 LLM 호출
        if base["clause_type"] == "other" or base["confidence"] < llm_conf_threshold:
            should_llm = True

    if should_llm:
        try:
            llm_out = llm_tag_solar_pro2(text, api_key=upstage_api_key)
            merged = llm_out
        except Exception as e:
            # 네트워크/API 일시 오류가 있어도 파이프라인은 계속 진행합니다.
            merged = base
            merged["notes"] = f"LLM failed: {type(e).__name__}"
    else:
        merged = base

    merged = validate_and_override(text, merged)
    return merged


def tag_chunks(
    chunks: List[Document],
    *,
    insurer_code: str = "unknown",
    product_code: str = "unknown",
    pdf_name: str = "unknown.pdf",
    doc_type: str = "terms",
    embedding_model: str = "solar-embedding-1-large",
    use_llm_when: str = "unknown_or_low_conf",
    llm_conf_threshold: float = 0.70,
) -> List[Document]:
    """
    chunk(Document) 리스트를 입력받아 태깅 메타데이터를 확장한 Document 리스트를 반환합니다.

    Args:
        chunks: splitter에서 생성된 Document 리스트
        insurer_code/product_code/pdf_name/doc_type: 문서 식별 메타데이터
        embedding_model: 이후 임베딩 단계에서 사용하는 모델명 기록용
        use_llm_when: "always" | "unknown_or_low_conf" | "never"
        llm_conf_threshold: LLM 호출 기준 confidence 임계값

    Returns:
        metadata가 풍부하게 추가된 List[Document]
    """
    # LLM 호출 가능성이 있는 모드에서는 API Key가 반드시 필요합니다.
    upstage_api_key = os.getenv("UPSTAGE_API_KEY", "")
    if use_llm_when != "never" and not upstage_api_key:
        raise ValueError("UPSTAGE_API_KEY is not set. Please check your .env file.")

    docs: List[Document] = []
    for idx, chunk in enumerate(chunks):
        # tag_chunk는 str 입력을 기대하므로 page_content만 전달합니다.
        chunk_text = chunk.page_content
        tag = tag_chunk(
            chunk_text,
            upstage_api_key=upstage_api_key,
            use_llm_when=use_llm_when,
            llm_conf_threshold=llm_conf_threshold,
        )

        # 페이지 메타는 upstream 로더/스플리터에 따라 없을 수 있으므로
        # 없는 경우 None으로 남기고 downstream에서 처리하도록 둡니다.
        page = chunk.metadata.get("page")
        page_start = chunk.metadata.get("page_start", page)
        page_end = chunk.metadata.get("page_end", page)

        # 최종 metadata 스키마:
        # - doc: 문서 식별 정보
        # - clause: 태깅 결과
        # - indexing: 검색/추적용 운영 메타
        # - source_metadata: 원본 chunk의 metadata 보존본
        metadata = {
            "doc": {
                "insurer_code": insurer_code,
                "product_code": product_code,
                "doc_type": doc_type,
                "pdf_name": pdf_name,
                "page_start": page_start,
                "page_end": page_end,
            },
            "clause": {
                "clause_type": tag["clause_type"],
                "risk_domains": tag["risk_domains"],
            },
            "indexing": {
                "chunk_id": f"chunk_{idx:06d}",
                "chunk_char_len": len(chunk_text),
                "embedding_model": embedding_model,
                "tag_method": tag["method"],
                "tag_confidence": tag["confidence"],
            },
            "source_metadata": chunk.metadata,
        }

        # page_content는 원문 텍스트를 유지하고 metadata만 확장합니다.
        docs.append(Document(page_content=chunk_text, metadata=metadata))

    return docs
