from datetime import datetime
from pathlib import Path
from typing import Literal

from langchain.chat_models import init_chat_model
from langchain_core.documents import Document
from pydantic import BaseModel, Field

from rich import print as rprint

from app.agents.rag_agent.constants import TOP_K
from app.agents.rag_agent.state.rag_state import RagState


def _document_key(document: Document) -> tuple[str, tuple[tuple[str, str], ...]]:
    metadata_items = tuple(
        sorted(
            (str(key), str(value)) for key, value in (document.metadata or {}).items()
        )
    )
    return document.page_content, metadata_items


def _evaluation_total_score(document: Document) -> int:
    evaluation = (document.metadata or {}).get("evaluation") or {}
    return int(evaluation.get("total_score", 0))


def _make_output_dir(timestamp: str) -> Path:
    rag_agent_dir = Path(__file__).resolve().parents[1]
    target_dir = rag_agent_dir / "data" / "retrieval" / timestamp
    target_dir.mkdir(parents=True, exist_ok=True)
    return target_dir


def _save_documents_as_python(
    output_file: Path, variable_name: str, documents: list[Document]
) -> None:
    lines: list[str] = [
        "from langchain_core.documents import Document",
        "",
        f"{variable_name}: list[Document] = [",
    ]

    for document in documents:
        lines.append(
            f"    Document(page_content={document.page_content!r}, metadata={dict(document.metadata or {})!r}),"
        )

    lines.append("]")
    lines.append("")
    output_file.write_text("\n".join(lines), encoding="utf-8")


class RelevanceEvaluationItem(BaseModel):
    doc_index: int = Field(description="평가 대상 문서 번호(1-based)")
    keyword_match_score: int = Field(
        ge=0, le=40, description="질의 핵심 키워드 일치도 (0~40)"
    )
    condition_fit_score: int = Field(
        ge=0, le=40, description="건강 조건(자주 아픈 부위/질병명) 부합도 (0~40)"
    )
    policy_actionability_score: int = Field(
        ge=0, le=20, description="보험 설계/추천 실무 활용도 (0~20)"
    )
    total_score: int = Field(ge=0, le=100, description="총 연관도 점수 (0~100)")
    judgement: Literal["high", "medium", "low"] = Field(description="정성 판단 레이블")
    reason: str = Field(description="점수 근거 요약")


class RelevanceEvaluationOutput(BaseModel):
    evaluations: list[RelevanceEvaluationItem] = Field(default_factory=list)


def _build_query_context(state: RagState) -> str:
    query_texts = [
        query for query in (state.query_texts or []) if query and query.strip()
    ]
    if query_texts:
        return "\n".join(f"- {query.strip()}" for query in query_texts)
    return ""


def _collect_query_texts(state: RagState) -> list[str]:
    query_texts = [
        query.strip() for query in (state.query_texts or []) if query and query.strip()
    ]
    if query_texts:
        return query_texts
    return []


def _split_docs_by_query(
    documents: list[Document], query_count: int, k: int = TOP_K
) -> list[list[Document]]:
    grouped: list[list[Document]] = []
    for query_index in range(query_count):
        start = query_index * k
        end = start + k
        grouped.append(documents[start:end])
    return grouped


def _evaluate_relevance_by_llm(
    query_context: str, documents: list[Document]
) -> dict[int, RelevanceEvaluationItem]:
    if not query_context or not documents:
        return {}

    document_blocks: list[str] = []
    for index, document in enumerate(documents, start=1):
        preview = document.page_content.replace("\n", " ").strip()[:1200]
        document_blocks.append(f"[문서 {index}]\n{preview}")

    prompt = f"""
너는 펫 보험 전문 보험설계사다.
아래 멀티 쿼리 컨텍스트와 검색 문서를 보고 문서별 연관도를 정량 평가해라.

[Query Context]
{query_context}

[평가 기준]
1) keyword_match_score (0~40): 질의 핵심 키워드/의도와 직접 매칭되는 정도
2) condition_fit_score (0~40): 건강 조건(자주 아픈 부위/질병명)에 맞는 정도
3) policy_actionability_score (0~20): 실제 보험 추천/설계에 바로 활용 가능한 정도
4) total_score는 반드시 위 3개 점수의 합으로 산정
5) judgement 규칙:
   - high: total_score >= 80
   - medium: 50 <= total_score < 80
   - low: total_score < 50

[문서 목록]
{chr(10).join(document_blocks)}

모든 문서를 빠짐없이 평가하라.
""".strip()

    model = init_chat_model(model="solar-pro2", temperature=0.0)
    structured_model = model.with_structured_output(RelevanceEvaluationOutput)
    result: RelevanceEvaluationOutput = structured_model.invoke(prompt)
    return {item.doc_index: item for item in result.evaluations}


def _attach_evaluation_metadata(
    documents: list[Document],
    evaluations: dict[int, RelevanceEvaluationItem],
    query_context: str,
    source: str,
) -> list[Document]:
    scored_docs: list[Document] = []

    for index, document in enumerate(documents, start=1):
        evaluation = evaluations.get(index)
        metadata = dict(document.metadata or {})
        if evaluation:
            metadata["evaluation"] = {
                "query_context": query_context,
                "source": source,
                "total_score": evaluation.total_score,
                "keyword_match_score": evaluation.keyword_match_score,
                "condition_fit_score": evaluation.condition_fit_score,
                "policy_actionability_score": evaluation.policy_actionability_score,
                "judgement": evaluation.judgement,
                "reason": evaluation.reason,
            }
        else:
            metadata["evaluation"] = {
                "query_context": query_context,
                "source": source,
                "total_score": 0,
                "keyword_match_score": 0,
                "condition_fit_score": 0,
                "policy_actionability_score": 0,
                "judgement": "low",
                "reason": "LLM 평가 결과 없음",
            }
        scored_docs.append(
            Document(page_content=document.page_content, metadata=metadata)
        )

    return scored_docs


def summary_multi(state: RagState) -> RagState:
    query_context = _build_query_context(state)
    query_texts = _collect_query_texts(state)
    normal_docs = state.terms_normal_tag_dense or []
    simple_docs = state.terms_simple_tag_dense or []

    normal_evaluations = _evaluate_relevance_by_llm(query_context, normal_docs)
    simple_evaluations = _evaluate_relevance_by_llm(query_context, simple_docs)

    normal_scored_docs = _attach_evaluation_metadata(
        documents=normal_docs,
        evaluations=normal_evaluations,
        query_context=query_context,
        source="terms_normal_tag_dense",
    )
    simple_scored_docs = _attach_evaluation_metadata(
        documents=simple_docs,
        evaluations=simple_evaluations,
        query_context=query_context,
        source="terms_simple_tag_dense",
    )
    normal_scored_docs.sort(key=_evaluation_total_score, reverse=True)
    simple_scored_docs.sort(key=_evaluation_total_score, reverse=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = _make_output_dir(timestamp)
    normal_docs_by_query = _split_docs_by_query(normal_docs, len(query_texts), TOP_K)
    simple_docs_by_query = _split_docs_by_query(simple_docs, len(query_texts), TOP_K)

    # query_text별로 normal/simple 2개 파일 생성 (각 파일은 k=TOP_K 기준 독립 결과 저장)
    for query_index, query_text in enumerate(query_texts, start=1):
        per_query_normal_source_docs = normal_docs_by_query[query_index - 1]
        per_query_simple_source_docs = simple_docs_by_query[query_index - 1]

        per_query_normal_evaluations = _evaluate_relevance_by_llm(
            query_text, per_query_normal_source_docs
        )
        per_query_simple_evaluations = _evaluate_relevance_by_llm(
            query_text, per_query_simple_source_docs
        )

        per_query_normal_docs = _attach_evaluation_metadata(
            documents=per_query_normal_source_docs,
            evaluations=per_query_normal_evaluations,
            query_context=query_text,
            source="terms_normal_tag_dense",
        )
        per_query_simple_docs = _attach_evaluation_metadata(
            documents=per_query_simple_source_docs,
            evaluations=per_query_simple_evaluations,
            query_context=query_text,
            source="terms_simple_tag_dense",
        )
        per_query_normal_docs.sort(key=_evaluation_total_score, reverse=True)
        per_query_simple_docs.sort(key=_evaluation_total_score, reverse=True)

        if per_query_normal_docs:
            _save_documents_as_python(
                output_file=output_dir
                / f"terms_normal_tag_dense_q{query_index}_{timestamp}.py",
                variable_name=f"terms_normal_tag_dense_q{query_index}_documents",
                documents=per_query_normal_docs,
            )
        if per_query_simple_docs:
            _save_documents_as_python(
                output_file=output_dir
                / f"terms_simple_tag_dense_q{query_index}_{timestamp}.py",
                variable_name=f"terms_simple_tag_dense_q{query_index}_documents",
                documents=per_query_simple_docs,
            )

    merged_docs: list[Document] = [*normal_scored_docs, *simple_scored_docs]

    merged_docs.sort(key=_evaluation_total_score, reverse=True)
    rprint("📝total dense retrieval results:", len(merged_docs))

    return {"retrieved_documents": merged_docs}
