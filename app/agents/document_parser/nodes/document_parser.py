import argparse
import runpy
import threading
import time

from dotenv import load_dotenv
from typing import List

from langchain_core.documents import Document
from langchain_upstage.document_parse import OutputFormat
from langchain_upstage import UpstageDocumentParseLoader

from rich import print as rprint

from app.agents.document_parser.nodes.path_utils import (
    DocumentParsePaths,
    build_document_parse_paths,
)
from app.agents.document_parser.state.document_parser_state import DocumentParserState


load_dotenv()


def document_parser_node(state: DocumentParserState):
    parse_document(file_name=state.file_name, output_format=state.output_format)


def parse_document(
    *,
    file_name: str,
    output_format: OutputFormat,
) -> List[Document]:
    """
    Document Parser를 이용해 원본 문서를 parsing하고 결과를 반환한다.
    """

    paths = build_document_parse_paths(file_name=file_name, output_format=output_format)
    # rprint("🔗 parse_document term_file_path:", paths.source_file_path)

    cached_result = _load_existing_dp_result(paths=paths)
    if cached_result:
        rprint(
            f"♻️ document parser: found existing parsing result. skip parsing and load existing result. {len(cached_result)}"
        )
        return cached_result

    dp_loader = UpstageDocumentParseLoader(
        file_path=str(paths.source_file_path),
        output_format=paths.dp_output_format,
        coordinates=False,
    )

    rprint("🚀 document parsing start !")
    start_time = time.perf_counter()
    stop_event = threading.Event()

    def _print_progress():
        while not stop_event.wait(5):
            elapsed = time.perf_counter() - start_time
            rprint(f"⏳ document parsing in progress... (elapsed: {elapsed:.2f}s)")

    progress_thread = threading.Thread(target=_print_progress, daemon=True)
    progress_thread.start()

    try:
        dp_result = dp_loader.load()
        _create_dp_result_file(
            dp_result=dp_result,
            paths=paths,
        )
    finally:
        stop_event.set()
        progress_thread.join()

    elapsed = time.perf_counter() - start_time
    rprint(
        f"✅ document parsing done. result length: {len(dp_result)} (elapsed: {elapsed:.2f}s)"
    )

    if not dp_result:
        raise ValueError("❗️ invalid document parsing result")

    return dp_result


def _load_existing_dp_result(*, paths: DocumentParsePaths) -> List[Document] | None:
    if not paths.dp_result_file_path.exists():
        return None

    try:
        namespace = runpy.run_path(str(paths.dp_result_file_path))
    except Exception as exc:
        rprint(
            f"⚠️ failed to load cached dp_result file: {paths.dp_result_file_path} ({exc})"
        )
        return None

    cached_output_format = namespace.get("dp_output_format")
    if cached_output_format and cached_output_format != paths.dp_output_format:
        rprint(
            "⚠️ cached output format mismatch. "
            f"cached={cached_output_format}, requested={paths.dp_output_format}. re-parse document."
        )
        return None

    dp_result = namespace.get("dp_result")
    if not isinstance(dp_result, list) or not all(
        isinstance(doc, Document) for doc in dp_result
    ):
        rprint(f"⚠️ invalid cached dp_result format: {paths.dp_result_file_path}")
        return None
    if not dp_result:
        rprint(
            f"⚠️ empty cached dp_result. re-parse document: {paths.dp_result_file_path}"
        )
        return None

    return dp_result


def _create_dp_result_file(*, dp_result: List[Document], paths: DocumentParsePaths):
    paths.source_file_dir.mkdir(parents=True, exist_ok=True)
    paths.dp_output_dir.mkdir(parents=True, exist_ok=True)

    lines = [
        "from langchain_core.documents import Document",
        "",
        f"dp_output_format = {paths.dp_output_format!r}",
        "",
        "dp_result = [",
    ]
    for doc in dp_result:
        lines.append(
            f"    Document(page_content={doc.page_content!r}, metadata={doc.metadata!r}),"
        )
    lines.append("]")
    lines.append("")

    paths.dp_result_file_path.write_text("\n".join(lines), encoding="utf-8")
    # rprint(f"✅ saved dp_result python file: {paths.dp_result_file_path}")

    _save_parsed_documents(
        dp_result=dp_result,
        paths=paths,
    )


def _save_parsed_documents(
    *,
    dp_result: List[Document],
    paths: DocumentParsePaths,
) -> None:
    paths.dp_output_dir.mkdir(parents=True, exist_ok=True)

    for idx, doc in enumerate(dp_result, start=1):
        if len(dp_result) == 1:
            output_file_path = (
                paths.dp_output_dir / f"dp_result.{paths.dp_output_extension}"
            )
        else:
            output_file_path = (
                paths.dp_output_dir
                / f"dp_result_part_{idx}.{paths.dp_output_extension}"
            )

        output_file_path.write_text(doc.page_content, encoding="utf-8")
        # rprint(f"✅ saved parsed document: {output_file_path}")


def _create_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run document parser graph.")
    parser.add_argument(
        "--file-name",
        required=True,
        help="source file name",
    )
    parser.add_argument(
        "--output-format",
        required=True,
        help="document parsing output format",
    )
    return parser


if __name__ == "__main__":
    args = _create_arg_parser().parse_args()
    file_name = args.file_name
    output_format = args.output_format
    parse_document(file_name=file_name, output_format=output_format)


# uv run python -m app.agents.document_parser.nodes.document_parser --file-name meritz_1_maum_pet_1_21_22_50.pdf --output-format markdown
