from app.agents.rag_agent.nodes.document_parse import parse_document
from app.agents.rag_agent.nodes.splitter import split_dp_result
from app.agents.rag_agent.nodes.vector_store import ingest_chunks

if __name__ == "__main__":
    FILE_NAME = "meritz_terms_normal_1_5.pdf"

    dp_result = parse_document(FILE_NAME)

    chunks = split_dp_result(dp_result)

    ingest_chunks(FILE_NAME, chunks)
