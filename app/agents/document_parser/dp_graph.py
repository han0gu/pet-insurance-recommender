from app.agents.document_parser.nodes.document_parse import parse_document
from app.agents.document_parser.nodes.splitter import split_dp_result
from app.agents.document_parser.nodes.vector_store import ingest_chunks

FILE_NAME = "meritz_terms_normal_1_5.pdf"

if __name__ == "__main__":
    dp_result = parse_document(FILE_NAME)

    chunks = split_dp_result(dp_result)

    ingest_chunks(FILE_NAME, chunks)


# uv run python -m app.agents.document_parser.dp_graph
