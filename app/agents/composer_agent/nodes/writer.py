from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_upstage import ChatUpstage
from app.agents.judge_agent.state import JudgeAgentState
from app.agents.vet_agent.state import VetAgentState


# .env 파일 로드 (API Key 때문에 필수)
load_dotenv()

def writer_node(state: JudgeAgentState):

    # 1. 데이터 꺼내기
    vet_field_keys = VetAgentState.model_fields.keys()
    vet_data = state.model_dump(include=vet_field_keys)    # 유저/강아지 정보 (이름, 견종 등)
    val_result = state.validation_result # 검증 결과 (점수, 이유)
    
    # 2. LLM 설정 (창의적인 글쓰기를 위해 temperature를 약간 높임)
    llm = ChatUpstage(model="solar-pro2", temperature=0.7)
    
    # 3. 프롬프트 작성
    system_prompt = """당신은 다정하고 전문적인 '펫보험 상담사'입니다.
    고객의 강아지 정보를 바탕으로, [검증 결과]를 알기 쉽게 설명해주는 편지를 작성하세요.
    
    # 작성 가이드
    1. 강아지(품종, 나이)를 걱정하는 따뜻한 말투로 시작하세요.
    2. **가장 추천하는 상품(1순위)**을 명확히 제시하고, 왜 이 상품이 강아지에게 딱 맞는지 설명하세요.
    3. 점수가 낮은 다른 상품들이 왜 탈락했거나 후순위인지도 간략히 언급하세요(비교 분석).
    4. 마지막으로 "더 궁금한 점이 있으신가요?" 같은 멘트로 마무리하세요.
    """

    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human", """
        === [강아지 정보] ===
        {vet_data}
        
        === [검증 및 추천 결과] ===
        {val_result}
        """)
    ])

    # 4. 실행
    chain = prompt | llm | StrOutputParser() 
    
    final_msg = chain.invoke({
        "vet_data": str(vet_data),
        "val_result": str(val_result)
    })
    
    
    # 5. State 업데이트
    return {"final_message": final_msg}
