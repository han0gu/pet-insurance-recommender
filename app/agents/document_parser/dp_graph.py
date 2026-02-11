from app.agents.document_parser.nodes.document_parse import parse_document
from app.agents.document_parser.nodes.splitter import split_dp_result
from app.agents.document_parser.nodes.vector_store import ingest_chunks

COLLECTION_NAME = "pet-insurance-recommender-v1.0"

if __name__ == "__main__":
    FILE_NAME = "meritz_non_dividend_petpermint_cat_family_46_170.pdf"
    dp_result = parse_document(FILE_NAME)

    chunks = split_dp_result(dp_result)

    ingest_chunks(COLLECTION_NAME, chunks)


# uv run python -m app.agents.document_parser.dp_graph
