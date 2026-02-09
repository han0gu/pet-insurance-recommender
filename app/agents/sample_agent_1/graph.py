from app.agents.sample_agent_1.nodes.sample_node import generate_korean_sentence
from app.agents.sample_agent_1.state.sample_state import SampleAgent1State


def run_agent_1() -> SampleAgent1State:
    sentence = generate_korean_sentence()
    return SampleAgent1State(korean_sentence=sentence)
