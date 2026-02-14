import argparse

from langgraph.graph.state import StateGraph, CompiledStateGraph, START, END

from app.agents.document_parser.constants import COLLECTION_NAME
from app.agents.document_parser.nodes import document_parser
from app.agents.document_parser.nodes.splitter import page_splitter, text_splitter
from app.agents.document_parser.nodes.tagger import tagger, tagger_simple
from app.agents.document_parser.nodes import vector_store
from app.agents.document_parser.state.document_parser_state import DocumentParserState

from app.agents import utils


def build_graph() -> CompiledStateGraph:
    workflow = StateGraph(DocumentParserState)

    workflow.add_node("document_parser", document_parser.document_parser_node)

    workflow.add_edge(START, "document_parser")
    workflow.add_edge("document_parser", END)

    graph = workflow.compile()

    utils.create_graph_image(
        graph,
        utils.get_current_file_name(__file__, True),
        utils.get_parent_path(__file__),
    )

    return graph


def create_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run document parser graph.")
    parser.add_argument(
        "--file-name",
        help="PDF file name  with extension",
    )
    parser.add_argument(
        "--ingest",
        action="store_true",
        help="해당 옵션이 포함된 경우 Vector DB 적재까지 진행",
    )
    parser.add_argument(
        "--basic-term-start",
        type=int,
        required=True,
        help="보통약관 시작 페이지 번호(footer 기준)",
    )
    parser.add_argument(
        "--basic-term-end",
        type=int,
        required=True,
        help="보통약관 종료 페이지 번호(footer 기준)",
    )
    parser.add_argument(
        "--special-term-start",
        type=int,
        required=True,
        help="특별약관 시작 페이지 번호(footer 기준)",
    )
    parser.add_argument(
        "--special-term-end",
        type=int,
        required=True,
        help="특별약관 종료 페이지 번호(footer 기준)",
    )
    return parser


if __name__ == "__main__":
    args = create_arg_parser().parse_args()
    file_name: str = args.file_name

    dp_result = document_parser.parse_document(file_name)

    pages = page_splitter.split_pages_and_add_metadata(
        dp_result,
        file_name,
        basic_term_start=args.basic_term_start,
        basic_term_end=args.basic_term_end,
        special_term_start=args.special_term_start,
        special_term_end=args.special_term_end,
    )

    chunks = text_splitter.split(pages)

    tagged_chunks = tagger_simple.tag_chunks(chunks)

    if args.ingest:
        vector_store.ingest_chunks(COLLECTION_NAME, tagged_chunks)


# uv run python -m app.agents.document_parser.dp_graph --file-name meritz_1_maum_pet_12_61.pdf --basic-term-start 1 --basic-term-end 21 --special-term-start 22 --special-term-end 50
# uv run python -m app.agents.document_parser.dp_graph --file-name meritz_1_maum_pet_12_61.pdf --basic-term-start 1 --basic-term-end 21 --special-term-start 22 --special-term-end 50 --ingest
