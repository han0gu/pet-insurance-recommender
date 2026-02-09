from app.agents.sample_agent_1.state.sample_state import SampleAgent1State
from app.agents.sample_agent_2.nodes.sample_node import translate_to_english
from app.agents.sample_agent_2.state.sample_state import SampleAgent2State


def run_agent_2(agent_1_state: SampleAgent1State) -> SampleAgent2State:
    english_sentence = translate_to_english(agent_1_state.korean_sentence)
    return SampleAgent2State(
        korean_sentence=agent_1_state.korean_sentence,
        english_sentence=english_sentence,
    )
