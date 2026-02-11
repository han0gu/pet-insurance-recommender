from langchain.chat_models import init_chat_model

from rich import print as rprint

from app.agents.rag_agent.state.rag_state import RagState, GenerateUserQueryOutput

from app.agents.vet_agent.state import VetAgentState


def generate_user_query(state: VetAgentState) -> RagState:
    # rprint(">>> generate_user_query input state", state)

    if not state or not state.species:
        raise ValueError("invalid VetAgentState !")

    prompt = f"""
사용자의 정보를 알려줄게. 

종: {state.species}
품종: {state.breed}
나이: {state.age}
성별: {state.gender.name}
체중: {state.weight}

위 정보를 빠짐없이 모두 포함해서, 마치 사용자가 직접 보험 상품 추천을 요청한 것과 같은 자연스러운 질문 문장으로 만들어줘. 생성된 문장에는 사용자의 정보가 단 하나도 빠지지 않고 포함되어야 해.
"""
    # rprint(">>> generated prompt", prompt)

    MODEL = "solar-pro2"
    llm = init_chat_model(model=MODEL, temperature=0.0)
    structured_llm = llm.with_structured_output(GenerateUserQueryOutput)
    llm_response: GenerateUserQueryOutput = structured_llm.invoke(
        # [{"role": "system", "content": prompt}]
        prompt
    )
    # rprint(">>> llm_response", llm_response)

    return {"user_query": llm_response.user_query}
