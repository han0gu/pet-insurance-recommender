from langchain.chat_models import init_chat_model
from langchain_core.prompts import ChatPromptTemplate
from judge_agent.state import GraphState, ValidationResult

def validator_node(state: GraphState):
    print("\n [Validator] RAG 후보군 정밀 검증 시작 (Filtering & Ranking)")

    # 1. State에서 데이터 꺼내기 
    user_data = state['user_profile']