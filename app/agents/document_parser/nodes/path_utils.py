from dataclasses import dataclass
from pathlib import Path

from langchain_upstage.document_parse import OutputFormat

from app.agents.document_parser.constants import TAG_TYPE, TERMS_DIR


output_extension_by_format = {
    "html": "html",
    "text": "txt",
    "markdown": "md",
}


@dataclass(frozen=True)
class DocumentBasePaths:
    source_file_name: str
    source_file_stem: str
    source_file_dir: Path
    source_file_path: Path


@dataclass(frozen=True)
class DocumentParsePaths(DocumentBasePaths):
    dp_output_format: OutputFormat
    dp_output_extension: str
    dp_output_dir: Path
    dp_result_file_path: Path


def build_document_base_paths(*, file_name: str) -> DocumentBasePaths:
    stem = Path(file_name).stem
    term_file_dir = TERMS_DIR / stem
    return DocumentBasePaths(
        source_file_name=file_name,
        source_file_stem=stem,
        source_file_dir=term_file_dir,
        source_file_path=term_file_dir / file_name,
    )


def build_document_parse_paths(
    *, file_name: str, output_format: OutputFormat
) -> DocumentParsePaths:
    output_extension = output_extension_by_format.get(output_format)
    if not output_extension:
        raise ValueError(f"❗️ unsupported output_format: {output_format}")

    base_paths = build_document_base_paths(file_name=file_name)
    return DocumentParsePaths(
        # base
        source_file_name=base_paths.source_file_name,
        source_file_stem=base_paths.source_file_stem,
        source_file_dir=base_paths.source_file_dir,
        source_file_path=base_paths.source_file_path,
        # others
        dp_output_format=output_format,
        dp_output_extension=output_extension,
        dp_output_dir=base_paths.source_file_dir / output_format,
        dp_result_file_path=base_paths.source_file_dir / output_format / "dp_result.py",
    )


def build_chunks_dir(*, file_name: str, output_format: OutputFormat) -> Path:
    paths = build_document_parse_paths(
        file_name=file_name,
        output_format=output_format,
    )
    return paths.dp_output_dir / "chunks"


def build_tagged_chunks_dir(
    *, file_name: str, output_format: OutputFormat, tag_type: TAG_TYPE
) -> Path:
    paths = build_document_parse_paths(
        file_name=file_name,
        output_format=output_format,
    )
    return paths.dp_output_dir / "tagged_chunks" / tag_type
