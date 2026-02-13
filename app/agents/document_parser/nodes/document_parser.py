import argparse

from pathlib import Path
from typing import List

from dotenv import load_dotenv

from langchain_core.documents import Document
from langchain_upstage.document_parse import OutputFormat
from langchain_upstage import UpstageDocumentParseLoader

from rich import print as rprint

from app.agents.document_parser.constants import TERMS_DIR
from app.agents.document_parser.nodes.splitter import page_splitter

output_extension_by_format = {
    "html": "html",
    "text": "txt",
    "markdown": "md",
}

load_dotenv()


def parse_document(
    file_name: str, output_format: OutputFormat = "html"
) -> List[Document]:
    TERM_FILE_PATH = TERMS_DIR / file_name
    rprint("🔗parse_document TERM_FILE_PATH:", TERM_FILE_PATH)

    dp_loader = UpstageDocumentParseLoader(
        file_path=str(TERM_FILE_PATH),
        output_format=output_format,
        coordinates=False,
    )

    rprint("🚀document parsing start. output format:", output_format)
    dp_result = (
        dp_loader.load()
    )  # UpstageDocumentParse는 전체 문서를 하나의 Document로 추출함
    rprint("✅document parsing done. result length:", len(dp_result))  # 1

    dp_split_result: List[Document] = []
    if len(dp_result) == 1 and "page" not in dp_result[0].metadata:
        dp_split_result = page_splitter.split_pages_and_add_metadata(
            dp_result[0], file_name
        )

    for doc in dp_split_result:
        DP_RESULTS_DIR = TERMS_DIR / file_name.split(".")[0]
        # rprint("🔗parse_document DP_RESULTS_DIR:", DP_RESULTS_DIR)
        create_page_content_file(doc, DP_RESULTS_DIR, output_format)
        # create_metadata_file(doc, DP_RESULTS_DIR)

    return dp_split_result


def create_page_content_file(
    doc: Document,
    target_dir: Path,
    output_format: OutputFormat = "html",
):
    target_dir.mkdir(parents=True, exist_ok=True)

    file_name_without_extension = doc.metadata["doc"]["file_name"].split(".")[0]
    OUTPUT_EXTENSION = output_extension_by_format.get(output_format)
    OUTPUT_FILE_NAME = f"{file_name_without_extension}_{doc.metadata['doc']['page']}.{OUTPUT_EXTENSION}"
    OUTPUT_FILE_PATH = target_dir / OUTPUT_FILE_NAME
    # rprint("🔗create_local_file OUTPUT_FILE_PATH:", OUTPUT_FILE_PATH)
    if OUTPUT_FILE_PATH.exists():
        # rprint("⚠️ create_local_file skipped (already exists)")
        return

    # rprint("🚀create_local_file start")
    OUTPUT_FILE_PATH.write_text(doc.page_content, encoding="utf-8")
    # rprint("✅create_local_file done")


def create_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run document parser graph.")
    parser.add_argument(
        "--file-name",
        help="PDF file name under app/agents/document_parser/data/terms",
    )
    return parser


if __name__ == "__main__":
    args = create_arg_parser().parse_args()
    file_name = args.file_name
    parse_document(file_name)


# uv run python -m app.agents.document_parser.nodes.document_parser --file-name meritz_1_maum_pet_12_16.pdf
