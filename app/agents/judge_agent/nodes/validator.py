import os
from app.agents.vet_agent.state.vet_state import VetAgentState
from dotenv import load_dotenv
from langchain_upstage import ChatUpstage 
from langchain_core.prompts import ChatPromptTemplate
from ..state import JudgeAgentState, ValidationResult

from rich import print as rprint

# .env 파일 로드 (API Key 때문에 필수)
load_dotenv()

# ==========================================
# 검증 노드 핵심 로직
# ==========================================
def validator_node(state: JudgeAgentState):

    # 1. 데이터 꺼내기 
    vet_field_keys = VetAgentState.model_fields.keys()
    vet_data = state.model_dump(include=vet_field_keys)
    docs = state.retrieved_documents

    # 2. Documents 객체들을 LLM이 읽을 수 있는 문자열로 변환 
    # (실제 RAG에서는 page_content에 약관 텍스트가 있음)
    rag_context = ""
    for idx, doc in enumerate(docs):
        rag_context += f"\n[약관 {idx+1}] {doc.page_content}\n"

    # 3. LLM 설정 
    llm = ChatUpstage(model="solar-pro2", temperature=0)
    structured_llm = llm.with_structured_output(ValidationResult)

    # 4. 프롬프트 수정 (User + Vet 정보가 하나로 합쳐짐)
    system_prompt = """당신은 보험 약관 심사 전문가입니다.
    제공된 [Vet Analysis]와 [Insurance Policies]를 비교하여 가입 적합성을 판단하세요.
    
    # 심사 기준 (엄격 준수)
    1. **질병 면책 (최우선 순위)**
        - 수의사가 진단한 질병(diseases)이 약관의 '보상하지 않는 손해(면책 사항)'에 포함되는지 글자 그대로 확인하세요.

    2. **나이 제한 (만 19세 기준)**
        - 이 보험은 **만 19세까지** 재가입이 가능합니다.
        - 반려동물의 나이가 **19세를 초과하는 경우에만** 나이를 거절 사유로 언급하세요.
        - 19세 이하(예: 10세, 15세 등)라면 나이 때문에 가입이 어렵다는 말을 절대 하지 마세요.

    3. **축종(Species) 확인**
        - 대상이 '개(Dog)'인지 '고양이(Cat)'인지만 확인하여 약관을 적용하세요.
    """        
    # - 약관에 명시되지 않은 질병은 '보장 가능'으로 간주하세요.

    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human", """
        === [Vet Analysis (유저 정보 + 진단 결과)] ===
        {vet_data}
        
        === [Insurance Policies (약관 검색 결과)] ===
        {rag_context}
        """)
    ])

    # 5. 실행
    # vet_data는 딕셔너리이므로 str()로 변환해서 주입
    chain = prompt | structured_llm
    result = chain.invoke({
        "vet_data": str(vet_data), 
        "rag_context": rag_context
    })
    
    
    return {"validation_result": result.model_dump()}
