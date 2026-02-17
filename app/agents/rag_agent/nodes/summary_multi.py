from datetime import datetime
from pathlib import Path
from pprint import pformat
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


def _save_literal_as_python(
    output_file: Path, variable_name: str, value: object
) -> None:
    lines = [f"{variable_name} = {pformat(value, sort_dicts=False)}", ""]
    output_file.write_text("\n".join(lines), encoding="utf-8")


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


def _sorted_total_scores(
    evaluations: dict[int, RelevanceEvaluationItem],
) -> list[int]:
    return [
        item.total_score
        for _, item in sorted(evaluations.items(), key=lambda entry: entry[0])
    ]


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


def _average_total_score(total_scores: list[int]) -> float:
    if not total_scores:
        return 0.0
    return round(sum(total_scores) / len(total_scores), 2)


def _process_collection_with_unfiltered(
    *,
    query_context: str,
    query_texts: list[str],
    filtered_docs: list[Document],
    unfiltered_docs: list[Document],
    source_name: str,
    output_dir: Path,
    timestamp: str,
) -> tuple[list[Document], list[dict[str, object]]]:
    filtered_evaluations = _evaluate_relevance_by_llm(query_context, filtered_docs)
    filtered_scored_docs = _attach_evaluation_metadata(
        documents=filtered_docs,
        evaluations=filtered_evaluations,
        query_context=query_context,
        source=source_name,
    )
    filtered_scored_docs.sort(key=_evaluation_total_score, reverse=True)

    filtered_docs_by_query = _split_docs_by_query(filtered_docs, len(query_texts), TOP_K)
    unfiltered_docs_by_query = _split_docs_by_query(
        unfiltered_docs, len(query_texts), TOP_K
    )
    comparison_rows: list[dict[str, object]] = []
    unfiltered_source_name = f"{source_name}_unfiltered"

    for query_index, query_text in enumerate(query_texts, start=1):
        per_query_filtered_docs = filtered_docs_by_query[query_index - 1]
        per_query_unfiltered_docs = unfiltered_docs_by_query[query_index - 1]

        per_query_filtered_evaluations = _evaluate_relevance_by_llm(
            query_text, per_query_filtered_docs
        )
        per_query_unfiltered_evaluations = _evaluate_relevance_by_llm(
            query_text, per_query_unfiltered_docs
        )

        per_query_filtered_scored_docs = _attach_evaluation_metadata(
            documents=per_query_filtered_docs,
            evaluations=per_query_filtered_evaluations,
            query_context=query_text,
            source=source_name,
        )
        per_query_unfiltered_scored_docs = _attach_evaluation_metadata(
            documents=per_query_unfiltered_docs,
            evaluations=per_query_unfiltered_evaluations,
            query_context=query_text,
            source=unfiltered_source_name,
        )
        per_query_filtered_scored_docs.sort(key=_evaluation_total_score, reverse=True)
        per_query_unfiltered_scored_docs.sort(key=_evaluation_total_score, reverse=True)

        if per_query_filtered_scored_docs:
            _save_documents_as_python(
                output_file=output_dir / f"{source_name}_q{query_index}_{timestamp}.py",
                variable_name=f"{source_name}_q{query_index}_documents",
                documents=per_query_filtered_scored_docs,
            )
        if per_query_unfiltered_scored_docs:
            _save_documents_as_python(
                output_file=output_dir
                / f"{unfiltered_source_name}_q{query_index}_{timestamp}.py",
                variable_name=f"{unfiltered_source_name}_q{query_index}_documents",
                documents=per_query_unfiltered_scored_docs,
            )

        filtered_total_scores = _sorted_total_scores(per_query_filtered_evaluations)
        unfiltered_total_scores = _sorted_total_scores(per_query_unfiltered_evaluations)
        filtered_avg = _average_total_score(filtered_total_scores)
        unfiltered_avg = _average_total_score(unfiltered_total_scores)
        comparison_rows.append(
            {
                "query_index": query_index,
                "query_text": query_text,
                "filtered_total_scores": filtered_total_scores,
                "unfiltered_total_scores": unfiltered_total_scores,
                "filtered_avg_total_score": filtered_avg,
                "unfiltered_avg_total_score": unfiltered_avg,
                "avg_total_score_delta": round(filtered_avg - unfiltered_avg, 2),
            }
        )

    return filtered_scored_docs, comparison_rows


def summary_multi(state: RagState) -> RagState:
    query_context = _build_query_context(state)
    query_texts = _collect_query_texts(state)
    normal_docs = state.terms_normal_tag_dense or []
    normal_docs_unfiltered = state.terms_normal_tag_dense_unfiltered or []
    simple_docs = state.terms_simple_tag_dense or []
    simple_docs_unfiltered = state.terms_simple_tag_dense_unfiltered or []

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = _make_output_dir(timestamp)
    normal_scored_docs, normal_total_score_comparison_rows = (
        _process_collection_with_unfiltered(
            query_context=query_context,
            query_texts=query_texts,
            filtered_docs=normal_docs,
            unfiltered_docs=normal_docs_unfiltered,
            source_name="terms_normal_tag_dense",
            output_dir=output_dir,
            timestamp=timestamp,
        )
    )
    simple_scored_docs, simple_total_score_comparison_rows = (
        _process_collection_with_unfiltered(
            query_context=query_context,
            query_texts=query_texts,
            filtered_docs=simple_docs,
            unfiltered_docs=simple_docs_unfiltered,
            source_name="terms_simple_tag_dense",
            output_dir=output_dir,
            timestamp=timestamp,
        )
    )

    if normal_total_score_comparison_rows:
        _save_literal_as_python(
            output_file=output_dir
            / f"terms_normal_total_score_comparison_{timestamp}.py",
            variable_name="terms_normal_total_score_comparison",
            value=normal_total_score_comparison_rows,
        )

    if simple_total_score_comparison_rows:
        _save_literal_as_python(
            output_file=output_dir
            / f"terms_simple_total_score_comparison_{timestamp}.py",
            variable_name="terms_simple_total_score_comparison",
            value=simple_total_score_comparison_rows,
        )

    merged_docs: list[Document] = [*normal_scored_docs, *simple_scored_docs]

    merged_docs.sort(key=_evaluation_total_score, reverse=True)
    rprint("📝total dense retrieval results:", len(merged_docs))

    return {"retrieved_documents": merged_docs}
