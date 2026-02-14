from __future__ import annotations

from pathlib import Path
from pprint import pformat

from langchain_core.documents import Document


def create_chunk_file(
    *,
    chunk: Document,
    target_dir: Path,
    overwrite: bool = True,
) -> None:
    target_dir.mkdir(parents=True, exist_ok=True)

    chunk_id = chunk.metadata["indexing"]["chunk_id"]
    chunk_id_only_number = chunk_id.split("_")[-1]
    output_file_name = f"{target_dir.parent.name}_{chunk_id_only_number}.py"
    output_file_path = target_dir / output_file_name
    if not overwrite and output_file_path.exists():
        return

    chunk_literal = pformat(
        {
            "page_content": chunk.page_content,
            "metadata": chunk.metadata,
        },
        sort_dicts=False,
    )
    output_file_path.write_text(f"chunk = {chunk_literal}\n", encoding="utf-8")
