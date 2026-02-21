from __future__ import annotations

import re
from runpy import run_path
from typing import List

from langchain_core.documents import Document
from langchain_upstage.document_parse import OutputFormat

from app.agents.document_parser.constants import TAG_TYPE
from app.agents.document_parser.nodes.path_utils import build_tagged_chunks_dir

_CHUNK_FILE_PATTERN = re.compile(r"^chunk_(\d{6})\.py$")


def load_chunk_files(
    *,
    file_name: str,
    output_format: OutputFormat,
    tag_type: TAG_TYPE,
) -> List[Document]:
    chunk_dir = build_tagged_chunks_dir(
        file_name=file_name,
        output_format=output_format,
        tag_type=tag_type,
    )
    if not chunk_dir.exists():
        return []

    chunk_files = sorted(
        (
            path
            for path in chunk_dir.glob("*.py")
            if _CHUNK_FILE_PATTERN.match(path.name)
        ),
        key=lambda path: int(_CHUNK_FILE_PATTERN.match(path.name).group(1)),
    )

    loaded_chunks: List[Document] = []
    for chunk_file in chunk_files:
        namespace = run_path(str(chunk_file))
        chunk_obj = namespace.get("chunk")
        if isinstance(chunk_obj, Document):
            loaded_chunks.append(chunk_obj)
            continue

        raise ValueError(
            f"Invalid chunk payload: {chunk_file}. Expected 'chunk' to be a Document."
        )

    return loaded_chunks
