import argparse
from pathlib import Path
from typing import List

from dotenv import load_dotenv

from langchain_core.documents import Document
from langchain_upstage.document_parse import OutputFormat
from langchain_upstage import UpstageDocumentParseLoader

from rich import print as rprint

load_dotenv()


def create_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run document parser graph.")
    parser.add_argument(
        "--file-name",
        help="PDF file name under app/agents/document_parser/data/terms",
    )
    return parser


def parse_document(
    file_name: str, output_format: OutputFormat = "html"
) -> List[Document]:
    # BASE_DIR = Path(os.getenv("PROJECT_ROOT", ".")).resolve() # project root
    BASE_DIR = Path(__file__).resolve().parent.parent  # app/agents/document_parser
    TERMS_DIR = BASE_DIR / "data" / "terms"
    TERM_FILE_PATH = TERMS_DIR / file_name
    print(">>> parse_document TERM_FILE_PATH", TERM_FILE_PATH)

    dp_loader = UpstageDocumentParseLoader(
        file_path=str(TERM_FILE_PATH),
        output_format=output_format,
        coordinates=False,
    )
    # rprint("dp_loader", dp_loader)

    dp_result = dp_loader.load()
    print("dp_result len", len(dp_result))  # 1
    create_local_file(dp_result, TERMS_DIR, file_name, output_format)

    return dp_result


def create_local_file(
    dp_result: List[Document],
    target_dir: Path,
    original_file_name: str,
    output_format: OutputFormat = "html",
):
    if not dp_result:
        raise ValueError("Document parse result is empty.")

    output_extension_by_format = {
        "html": "html",
        "text": "txt",
        "markdown": "md",
    }
    OUTPUT_EXTENSION = output_extension_by_format.get(output_format)
    if OUTPUT_EXTENSION is None:
        raise ValueError(f"Unsupported output_format: {output_format}")

    OUTPUT_FILE_NAME = f"{Path(original_file_name).stem}_dp_content.{OUTPUT_EXTENSION}"
    OUTPUT_FILE_PATH = target_dir / OUTPUT_FILE_NAME
    if OUTPUT_FILE_PATH.exists():
        print(">>> create_local_file skipped (already exists)", OUTPUT_FILE_PATH)
        return

    dp_content = dp_result[0].page_content
    # rprint(">>> dp_content\n", dp_content)
    OUTPUT_FILE_PATH.write_text(dp_content, encoding="utf-8")
    print(">>> create_local_file OUTPUT_FILE_PATH", OUTPUT_FILE_PATH)


if __name__ == "__main__":
    args = create_arg_parser().parse_args()
    file_name = args.file_name

    parse_document(file_name)
    # parse_document(file_name, "markdown")


# uv run python -m app.agents.document_parser.nodes.document_parse --file-name meritz_maum_pet_12_61.pdf
