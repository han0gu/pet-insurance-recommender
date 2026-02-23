from __future__ import annotations

from collections import Counter
from pathlib import Path
from pprint import pformat
from typing import Callable, Dict, Iterable, List

from langchain_core.documents import Document
from langchain_upstage.document_parse import OutputFormat

from app.agents.document_parser.constants import TAG_TYPE
from app.agents.document_parser.nodes.path_utils import build_tagged_chunks_dir


def summarize_counts(
    *,
    tagged_chunks: List[Document],
    tag_type: TAG_TYPE,
    output_format: OutputFormat,
    clause_types: Iterable[str],
    term_types: Iterable[str],
) -> Dict[str, Dict[str, int]]:
    clause_summary = summarize_clause_type_counts(
        tagged_chunks=tagged_chunks,
        clause_types=clause_types,
    )
    term_summary = summarize_term_type_counts(
        tagged_chunks=tagged_chunks,
        term_types=term_types,
    )
    combined_summary = {
        "clause_type": clause_summary,
        "term_type": term_summary,
    }

    if tagged_chunks:
        file_name = tagged_chunks[0].metadata["doc"]["file_name"]
        target_dir = build_tagged_chunks_dir(
            file_name=file_name,
            output_format=output_format,
            tag_type=tag_type,
        )
        create_combined_summary_file(
            summary=combined_summary,
            target_dir=target_dir,
        )

    return combined_summary


def summarize_term_type_counts(
    *,
    tagged_chunks: List[Document],
    term_types: Iterable[str],
) -> Dict[str, int]:
    return _summarize_label_counts(
        tagged_chunks=tagged_chunks,
        labels=term_types,
        value_getter=lambda chunk: chunk.metadata.get("term_type") or "other",
    )


def summarize_clause_type_counts(
    *,
    tagged_chunks: List[Document],
    clause_types: Iterable[str],
) -> Dict[str, int]:
    return _summarize_label_counts(
        tagged_chunks=tagged_chunks,
        labels=clause_types,
        value_getter=lambda chunk: chunk.metadata.get("clause", {}).get("clause_type"),
    )


def _summarize_label_counts(
    *,
    tagged_chunks: List[Document],
    labels: Iterable[str],
    value_getter: Callable[[Document], str | None],
) -> Dict[str, int]:
    if not tagged_chunks:
        return {"total": 0}

    summary: Dict[str, int] = {"total": len(tagged_chunks)}
    label_counter = Counter(
        value for value in (value_getter(chunk) for chunk in tagged_chunks) if value
    )

    for label in labels:
        summary[label] = label_counter.get(label, 0)

    for label, count in label_counter.items():
        if label not in summary:
            summary[label] = count

    return summary


def create_combined_summary_file(
    *,
    summary: Dict[str, Dict[str, int]],
    target_dir: Path,
) -> None:
    target_dir.mkdir(parents=True, exist_ok=True)

    output_file_name = "chunks_summary.py"
    output_file_path = target_dir / output_file_name
    output_file_path.parent.mkdir(parents=True, exist_ok=True)

    summary_literal = pformat(summary, sort_dicts=False)
    output_file_path.write_text(f"summary = {summary_literal}\n", encoding="utf-8")
