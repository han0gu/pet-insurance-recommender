from langchain_core.documents import Document
from rich import print as rprint

from app.agents.document_parser.nodes.embeddings import load_underlying_embeddings
from app.agents.document_parser.nodes.vector_store import setup_vector_store
from app.agents.rag_agent.constants import TOP_K
from app.agents.rag_agent.state.rag_state import RagState


def _document_key(document: Document) -> tuple[str, tuple[tuple[str, str], ...]]:
    metadata_items = tuple(
        sorted(
            (str(key), str(value)) for key, value in (document.metadata or {}).items()
        )
    )
    return document.page_content, metadata_items


def _collect_query_texts(state: RagState) -> list[str]:
    query_texts = [
        query.strip() for query in (state.query_texts or []) if query and query.strip()
    ]
    if query_texts:
        return query_texts
    raise ValueError("invalid query_texts !")


def _collect_embeddings(state: RagState) -> list[list[float]]:
    if state.query_texts_embeddings:
        return state.query_texts_embeddings
    return []


def _search_by_embeddings(
    collection_name: str, query_embeddings: list[list[float]], k: int = 3
) -> list[Document]:
    underlying_embeddings = load_underlying_embeddings()
    vector_store = setup_vector_store(
        underlying_embeddings=underlying_embeddings,
        collection_name=collection_name,
    )

    merged_docs: list[Document] = []

    for embedding in query_embeddings:
        docs = vector_store.similarity_search_by_vector(embedding, k=k)
        merged_docs.extend(docs)

    return merged_docs


def retrieve_normal_by_multi_query_texts(state: RagState) -> RagState:
    rprint("🔎normal tag collection: retrieve start")
    query_texts_embeddings = _collect_embeddings(state)
    if not query_texts_embeddings:
        query_texts = _collect_query_texts(state)
        query_texts_embeddings = load_underlying_embeddings().embed_documents(
            query_texts
        )
    search_result = _search_by_embeddings(
        "terms_normal_tag_dense", query_texts_embeddings, k=TOP_K
    )
    rprint(
        f"✅normal tag collection: retrieve complete (top_k={TOP_K}, results={len(search_result)})"
    )
    return {"terms_normal_tag_dense": search_result}


def retrieve_simple_by_multi_query_texts(state: RagState) -> RagState:
    rprint("🔎simple tag collection: retrieve start")
    query_texts_embeddings = _collect_embeddings(state)
    if not query_texts_embeddings:
        query_texts = _collect_query_texts(state)
        query_texts_embeddings = load_underlying_embeddings().embed_documents(
            query_texts
        )
    search_result = _search_by_embeddings(
        "terms_simple_tag_dense", query_texts_embeddings, k=TOP_K
    )
    rprint(
        f"✅simple tag collection: retrieve complete (top_k={TOP_K}, results={len(search_result)})"
    )
    return {"terms_simple_tag_dense": search_result}
