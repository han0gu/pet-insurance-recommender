from typing import TypedDict

from langgraph.graph import END, START, StateGraph

from app.agents.sample_agent_1.state.sample_state import SampleAgent1State
from app.agents.sample_agent_2.nodes.sample_node import translate_to_english
from app.agents.sample_agent_2.state.sample_state import SampleAgent2State


class Agent2GraphState(TypedDict, total=False):
    korean_sentence: str
    english_sentence: str


def _translate_node(state: Agent2GraphState) -> Agent2GraphState:
    korean_sentence = state["korean_sentence"]
    return {
        "korean_sentence": korean_sentence,
        "english_sentence": translate_to_english(korean_sentence),
    }


def build_agent_2_sub_graph():
    graph_builder = StateGraph(Agent2GraphState)
    graph_builder.add_node("translate_sentence", _translate_node)
    graph_builder.add_edge(START, "translate_sentence")
    graph_builder.add_edge("translate_sentence", END)
    return graph_builder.compile()


sample_agent_2_sub_graph = build_agent_2_sub_graph()


def run_agent_2(agent_1_state: SampleAgent1State) -> SampleAgent2State:
    result = sample_agent_2_sub_graph.invoke(
        {"korean_sentence": agent_1_state.korean_sentence}
    )
    return SampleAgent2State(
        korean_sentence=result["korean_sentence"],
        english_sentence=result["english_sentence"],
    )
