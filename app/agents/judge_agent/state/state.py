from enum import Enum
from typing import List, TypedDict, Optional
from pydantic import BaseModel, Field

"""
state 역할
유저 정보, 보험 정보, 검증 결가의 '데아터 규격을'을 정의하고 , 노드 간 주고 받을 상태(state)를 정의  
"""

# 1. 유저 & 수의사 관련 데이터 모델 
class Gender(str, Enum):
    mele = "mele"
    female = "female"

class DiseaseInfo(BaseModel):
    name: str = Field(description="질병명")
    incidence_rate: str = Field(description="발생률")
    onset_period: str = Field(description="발병 시기")

class VetAgentMockState(BaseModel):
    """유저가 입력한 반려동물 정보 (User Profile)"""
    species: str = Field(description="종")
    breed: str = Field(description="품종")
    age: int = Field(description="나이")
    gender: Gender = Field(description="성별")
    weight: float = Field(description="체중(kg)")
    diseases: list[DiseaseInfo] = Field(default_factory=list, description="질병 목록")

# 2. RAG 검색 결과 데이터 모델 (Input)
class InsurancePolicyMock(BaseModel):
    product_name: str = Field(description="보험 상품명")
    company_name: str = Field(description="보험사명")
    min_age: int = Field(description="가입 최소 나이")
    max_age: int = Field(description="가입 최대 나이")
    excluded_diseases: List[str] = Field(description="보장하지 않는 질병(면책 질병)")
    guaranteed_diseases: List[str] = Field(description="주요 보장 질병")

# 3. 검증 결과 데이터 모델 (Output)
class ValidatedPolicy(BaseModel):
    """검증을 통과한 보험 상품 정보 (이유 포함)"""
    product_name: str = Field(description="상품명")
    suitability_score: int = Field(description="적합성 점수 (1~100)")
    reason: str = Field(description="이 상품이 선정된 이유 (장점 위주)")
class ValidationResult(BaseModel):
    """검증 노드가 최종적으로 뱉을 결과물"""
    # Top 3개를 뽑아야 하므로 리스트 형태로 정의합니다.
    selected_policies: List[ValidatedPolicy] = Field(description="검증을 통과한 상위 3개 보험 상품")
    review_summary: str = Field(description="전체적인 검토 의견 (예: 5개 중 2개는 나이 제한으로 탈락했습니다.)")

# 4. LangGraph 상태 (state) 정의 
class GraphState(TypedDict):
    # 1. 고정값 (Input)
    user_profile: dict          # VetAgentMockState.model_dump()
    vet_response: str           # 수의사의 진단 내용 (Text)
    
    # 2. 가변값 (Process & Output)
    # RAG 검색 결과 (Top 5 후보군) - 리스트 형태임에 주의
    rag_candidates: List[dict]  # List[InsurancePolicyMock.model_dump()]
    
    # 검증 완료된 결과 (Top 3 선정작)
    validation_output: dict     # ValidationResult.model_dump()


