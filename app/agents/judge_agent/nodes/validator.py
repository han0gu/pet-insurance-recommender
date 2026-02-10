import os
from dotenv import load_dotenv
from langchain_upstage import ChatUpstage 
from langchain_core.prompts import ChatPromptTemplate
from ..state import AgentState, ValidationResult

# .env 파일 로드 (API Key 때문에 필수)
load_dotenv()

# ==========================================
# 검증 노드 핵심 로직
# ==========================================
def validator_node(state: AgentState):
    print("\n🚀 [Validator] Document 리스트 분석 및 검증 시작...")

    # 1. 데이터 꺼내기 
    vet_data = state.vet_result
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
    
    [Vet Analysis] 정보를 바탕으로 [Insurance Policies]를 검토하세요.
    - Vet Analysis에는 대상의 기본 정보(나이, 품종)와 **수의사가 진단한 질병 목록(diseases)**이 포함되어 있습니다.
    - 이 질병들이 보험의 면책 사항(보장하지 않는 질병)에 해당하는지 엄격히 확인하세요.
    - 나이 제한(age)도 반드시 확인해야 합니다.
    """

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
    
    print(f"✅ 검증 완료: {len(result.selected_policies)}개 상품 선정")
    
    return {"validation_result": result.model_dump()}
