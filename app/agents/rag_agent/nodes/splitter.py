from typing import List

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from rich import print as rprint


def load_splitter():
    CHUNK_SIZE = 300
    CHUNK_OVERLAP = 50
    CHUNK_SEPARATOR = ["\n\n", "\n", ". ", " ", ""]

    # TODO: markdown에 특화된 splitter?
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=CHUNK_SEPARATOR,
        strip_whitespace=True,
    )
    rprint("splitter", splitter)

    return splitter


def split_dp_result(dp_result: List[Document]) -> List[Document]:
    """
    split dp_result

    Args:
        dp_result

    Returns:
        chunks (List[Document]):
            - dp_result를 split한 결과
            - document["metadata"]["total_pages"]: 원본 문서(PDF)의 전체 페이지 수
            - len(chunks): 전체 문서를 chunk size와 chunk overlap을 기준으로 나눈 결과의 개수
    """
    splitter = load_splitter()

    chunks = splitter.split_documents(dp_result)
    print("chunks len", len(chunks))
    # rprint(">>> sample chunk", chunks[0])

    # 실습에서도 그렇고, 'page' 속성이 없으므로 무의미한 코드
    # page_counts: dict = Counter(chunk.metadata.get("page", 1) for chunk in chunks)
    # print("page_counts", page_counts)

    return chunks
