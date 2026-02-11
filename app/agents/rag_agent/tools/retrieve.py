from langchain_core.tools import tool

from rich import print as rprint

from app.agents.rag_agent.state.rag_state import RagState, RetrieveToolInput

from app.agents.document_parser.dp_graph import COLLECTION_NAME
from app.agents.document_parser.nodes.embeddings import load_underlying_embeddings
from app.agents.document_parser.nodes.vector_store import setup_vector_store


def retrieve(state: RagState) -> RagState:
    """
    tool 사용 시

    ```python
    @tool(
        args_schema=RetrieveToolInput,
        description="Vector DB에서 user_query_embedding와 유사한 내용 검색",
    )
    def retrieve(user_query_embedding: list[float]) -> RagState:
    ```
    """
    # rprint("retrieve input state", state)

    underlying_embeddings = load_underlying_embeddings()

    vector_store = setup_vector_store(
        underlying_embeddings=underlying_embeddings, collection_name=COLLECTION_NAME
    )

    search_result = vector_store.similarity_search_by_vector(
        state.user_query_embedding, k=3
    )
    # rprint(">>> search_result", [document.page_content for document in search_result])

    return {"retrieved_documents": search_result}
