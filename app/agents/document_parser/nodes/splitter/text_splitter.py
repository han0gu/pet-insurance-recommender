import re
from runpy import run_path
from typing import List

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_upstage.document_parse import OutputFormat

from rich import print as rprint

from app.agents.document_parser.nodes.path_utils import build_chunks_dir
from app.agents.document_parser.nodes.tagger.chunk_file import create_chunk_file

_CHUNK_FILE_PATTERN = re.compile(r"^chunk_(\d{6})\.py$")


def load_splitter():
    CHUNK_SIZE = 300
    CHUNK_OVERLAP = 50
    CHUNK_SEPARATOR = ["\n\n", "\n", ". ", " ", ""]

    # TODO: markdown에 특화된 splitter?
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=CHUNK_SEPARATOR,
        strip_whitespace=True,
    )
    # rprint(">>> splitter", splitter)

    return splitter


def split(*, dp_result: List[Document], output_format: OutputFormat) -> List[Document]:
    if not dp_result:
        return []

    file_name = dp_result[0].metadata["doc"]["file_name"]
    existing_chunks = _load_split_chunks(
        file_name=file_name,
        output_format=output_format,
    )
    if existing_chunks:
        rprint(
            f"♻️ text splitter: found existing chunking result. skip chunking and load existing chunks. {len(existing_chunks)}"
        )
        return existing_chunks

    splitter = load_splitter()

    rprint("🚀 split documents start")
    chunks = splitter.split_documents(dp_result)
    _save_split_chunks(chunks=chunks, output_format=output_format)
    rprint("✅ split documents done. chunks length:", len(chunks))
    # rprint(">>> sample chunk", chunks[0])

    return chunks


def _save_split_chunks(*, chunks: List[Document], output_format: OutputFormat) -> None:
    if not chunks:
        return

    file_name = chunks[0].metadata["doc"]["file_name"]
    target_dir = build_chunks_dir(
        file_name=file_name,
        output_format=output_format,
    )

    for idx, chunk in enumerate(chunks):
        create_chunk_file(
            chunk=chunk,
            target_dir=target_dir,
            output_file_name=f"chunk_{idx:06d}.py",
            overwrite=False,
        )


def _load_split_chunks(
    *, file_name: str, output_format: OutputFormat
) -> List[Document]:
    chunk_dir = build_chunks_dir(
        file_name=file_name,
        output_format=output_format,
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
            f"Invalid split chunk payload: {chunk_file}. Expected 'chunk' to be a Document."
        )

    return loaded_chunks
