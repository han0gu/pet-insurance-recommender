import argparse

from langchain_upstage.document_parse import OutputFormat
from langgraph.graph.state import StateGraph, CompiledStateGraph, START, END

from app.agents.document_parser.constants import (
    TAG_TYPE,
    VECTOR_STORE_NAME_BY_TAG_TYPE,
)
from app.agents.document_parser.nodes import document_parser
from app.agents.document_parser.nodes.path_utils import build_document_base_paths
from app.agents.document_parser.nodes.splitter import text_splitter
from app.agents.document_parser.nodes.splitter.metadata_adder import (
    add_metadata_to_documents,
)
from app.agents.document_parser.nodes.tagger import tagger
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
        required=True,
        help="PDF file name  with extension",
    )
    parser.add_argument(
        "--output-format",
        required=True,
        help="document parsing output format",
    )
    parser.add_argument(
        "--tag-type",
        required=True,
        help="metadata tag type",
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
    tag_type: TAG_TYPE = args.tag_type
    output_format: OutputFormat = args.output_format
    base_paths = build_document_base_paths(file_name=file_name)

    dp_result = document_parser.parse_document(
        file_name=base_paths.source_file_name,
        output_format=output_format,
    )

    dp_result_with_metadata = add_metadata_to_documents(
        documents=dp_result,
        file_name=file_name,
        basic_term_start=args.basic_term_start,
        basic_term_end=args.basic_term_end,
        special_term_start=args.special_term_start,
        special_term_end=args.special_term_end,
    )

    chunks = text_splitter.split(
        dp_result=dp_result_with_metadata,
        output_format=output_format,
    )

    tagged_chunks = tagger.tag_chunks(
        chunks=chunks,
        output_format=output_format,
        tag_type=tag_type,
    )

    if args.ingest:
        vector_store.ingest_chunks(
            VECTOR_STORE_NAME_BY_TAG_TYPE[tag_type],
            tagged_chunks,
        )


# uv run python -m app.agents.document_parser.dp_graph --file-name meritz_1_maum_pet_1_21_22_50.pdf --basic-term-start 1 --basic-term-end 21 --special-term-start 22 --special-term-end 50 --output-format markdown --tag-type simple
# uv run python -m app.agents.document_parser.dp_graph --file-name meritz_2_petpermint_cat_family_45_82_83_206.pdf --basic-term-start 45 --basic-term-end 82 --special-term-start 83 --special-term-end 206 --output-format markdown --tag-type simple
# uv run python -m app.agents.document_parser.dp_graph --file-name meritz_3_petpermint_dog_family_51_86_87_230.pdf --basic-term-start 51 --basic-term-end 86 --special-term-start 87 --special-term-end 230 --output-format markdown --tag-type simple
# uv run python -m app.agents.document_parser.dp_graph --file-name samsung_1_dog_anypet_3_20_21_47.pdf --basic-term-start 3 --basic-term-end 20 --special-term-start 21 --special-term-end 4 --output-format markdown --tag-type simple
# uv run python -m app.agents.document_parser.dp_graph --file-name samsung_2_cat_anypet_3_20_21_37.pdf --basic-term-start 3 --basic-term-end 20 --special-term-start 21 --special-term-end 37 --output-format markdown --tag-type simple
# uv run python -m app.agents.document_parser.dp_graph --file-name samsung_3_direct_good_pet_28_42_45_105.pdf --basic-term-start 28 --basic-term-end 42 --special-term-start 45 --special-term-end 105 --output-format markdown --tag-type simple
# uv run python -m app.agents.document_parser.dp_graph --file-name samsung_4_pet_majestic_dog_33_49_52_139.pdf --basic-term-start 33 --basic-term-end 49 --special-term-start 52 --special-term-end 139 --output-format markdown --tag-type simple
# uv run python -m app.agents.document_parser.dp_graph --file-name samsung_5_pet_majestic_cat_31_47_50_159.pdf --basic-term-start 31 --basic-term-end 47 --special-term-start 50 --special-term-end 159 --output-format markdown --tag-type simple
# uv run python -m app.agents.document_parser.dp_graph --file-name kb_17390_gold_pet_dog_54_72_74_166.pdf --basic-term-start 54 --basic-term-end 72 --special-term-start 74 --special-term-end 166 --output-format markdown --tag-type simple
