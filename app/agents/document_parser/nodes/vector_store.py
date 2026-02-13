import os
import time
from typing import List

from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams

from langchain_core.vectorstores import VectorStore
from langchain_core.documents import Document
from langchain_qdrant import QdrantVectorStore

from rich import print as rprint

from app.agents.document_parser.nodes.embeddings import load_underlying_embeddings

_global_vector_db_client = None


def setup_vector_store(
    underlying_embeddings,
    collection_name,
) -> VectorStore:
    global _global_vector_db_client

    qdrant_url = os.getenv("QDRANT_URL", "http://localhost:6333")
    # qdrant_api_key = os.getenv("QDRANT_API_KEY") # 현재 불필요, 추후에 필요할 수 있음
    vector_size = int(os.getenv("QDRANT_VECTOR_SIZE", "4096"))

    if _global_vector_db_client is None:
        _global_vector_db_client = QdrantClient(
            url=qdrant_url,
            # api_key=qdrant_api_key,
            timeout=30,
        )  # Docker Compose로 실행한 Qdrant 서버
        rprint("✅initialize vector store client", _global_vector_db_client)

    if not _global_vector_db_client.collection_exists(collection_name):
        _global_vector_db_client.create_collection(
            collection_name=collection_name,
            vectors_config=VectorParams(size=vector_size, distance=Distance.COSINE),
        )
        rprint("✅create vector store collection:", collection_name)

    """
    QdrantClient:
      - Qdrant의 Python SDK
      - Low-level DB client (Low-level REST/gRPC API)
      - Qdrant의 모든 기능 접근 가능
      - 직접 벡터 생성, payload 구성, upsert 포맷을 다 처리해야 함
      - DB 관리자 / 인프라 관점에서 유용
    QdrantVectorStore:
      - LangChain Document 기반 High-level wrapper (High-level API)
      - LangChain VectorStore interface의 Qdrant implementation
      - add_documents, similarity_search, retriever 변환 등 RAG pipeline과 바로 연결됨
      - AI application 개발 관점에서 유용
    Qdrant 세부 구조:
      - Point = row
      - Segment = 여러 row를 담은 data file + index, 검색 시 여러 Segment를 병렬로 검색
    """
    return QdrantVectorStore(
        client=_global_vector_db_client,
        collection_name=collection_name,
        embedding=underlying_embeddings,
    )


def ingest_chunks(collection_name: str, chunks: List[Document]) -> VectorStore:
    underlying_embeddings = load_underlying_embeddings()
    vector_store = setup_vector_store(underlying_embeddings, collection_name)

    if chunks:
        rprint("🚀ingest_chunks start")
        start_time = time.perf_counter()
        vector_store.add_documents(chunks)  # embedding + saving
        elapsed = time.perf_counter() - start_time
        rprint(f"✅ingest_chunks complete (elapsed: {elapsed:.2f}s)")
    else:
        rprint("⚠️no documents")

    return vector_store
