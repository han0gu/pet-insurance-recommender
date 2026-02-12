from pydantic import BaseModel, Field
from typing import Optional, List

from langchain_core.documents import Document


class RagState(BaseModel):
    user_query: Optional[str] | None = None
    user_query_embedding: Optional[List[float]] = Field(default_factory=list)
    retrieved_documents: Optional[List[Document]] = Field(default_factory=list)


class GenerateUserQueryOutput(BaseModel):
    user_query: str = Field(
        description="주어진 정보를 바탕으로 LLM이 생성한 사용자 질문"
    )


class RetrieveToolInput(RagState):
    user_query_embedding: List[float] = Field(
        ..., description="검색에 사용할 사용자 질의 임베딩 벡터"
    )
