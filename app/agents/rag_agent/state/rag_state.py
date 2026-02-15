from pydantic import BaseModel, Field
from typing import Optional, List

from langchain_core.documents import Document


class RagState(BaseModel):
    user_query: Optional[str] | None = Field(
        default=None,
        description="vector DB 검색 시 query로 사용하기 위해 LLM이 사용자 입력 정보를 바탕으로 생성한 문장",
    )
    user_query_embedding: Optional[List[float]] = Field(
        default_factory=list, description="user_query를 임베딩한 벡터 값"
    )
    terms_normal_tag_dense: Optional[List[Document]] = Field(
        default_factory=list,
        description="terms_normal_tag_dense 컬렉션에서 검색한 결과",
    )
    terms_simple_tag_dense: Optional[List[Document]] = Field(
        default_factory=list,
        description="terms_simple_tag_dense 컬렉션에서 검색한 결과",
    )
    retrieved_documents: Optional[List[Document]] = Field(
        default_factory=list,
        description="평가/정렬 후 최종 취합된 검색 결과",
    )


class GenerateUserQueryOutput(BaseModel):
    user_query: str = Field(
        description="주어진 정보를 바탕으로 LLM이 생성한 사용자 질문"
    )


class RetrieveToolInput(RagState):
    user_query_embedding: List[float] = Field(
        ..., description="검색에 사용할 사용자 질의 임베딩 벡터"
    )
