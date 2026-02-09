from app.agents.sample_agent_1.graph import run_agent_1
from app.agents.sample_agent_2.graph import run_agent_2


def run_test_orchestration() -> str:
    agent_1_state = run_agent_1()
    agent_2_state = run_agent_2(agent_1_state)
    return (
        f"최초 생성된 한글 문장은 {agent_1_state.korean_sentence}였으며, "
        f"번역된 결과는 {agent_2_state.english_sentence}입니다"
    )
