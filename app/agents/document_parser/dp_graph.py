import argparse

from app.agents.document_parser.nodes import document_parser
from app.agents.document_parser.nodes import splitter
from app.agents.document_parser.nodes import tagger
from app.agents.document_parser.nodes import vector_store

COLLECTION_NAME = "pet-insurance-recommender-v1.0"


def create_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run document parser graph.")
    parser.add_argument(
        "--file-name",
        help="PDF file name under app/agents/document_parser/data/terms",
    )
    parser.add_argument(
        "--ingest",
        action="store_true",
        help="해당 옵션이 포함된 경우 Vector DB 적재까지 진행",
    )
    return parser


if __name__ == "__main__":
    args = create_arg_parser().parse_args()
    file_name = args.file_name

    dp_result = document_parser.parse_document(file_name)

    chunks = splitter.split(dp_result)

    if args.ingest:
        tagged_chunks = tagger.tag_chunks(
            chunks,
            pdf_name=file_name,
        )

        vector_store.ingest_chunks(COLLECTION_NAME, tagged_chunks)


# uv run python -m app.agents.document_parser.dp_graph --file-name meritz_maum_pet_12_16.pdf
# uv run python -m app.agents.document_parser.dp_graph --file-name meritz_maum_pet_12_16.pdf --ingest
