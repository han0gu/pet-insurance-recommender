from typing import TypedDict

from langgraph.graph import END, START, StateGraph

from app.agents.sample_agent_1.nodes.sample_node import generate_korean_sentence
from app.agents.sample_agent_1.state.sample_state import SampleAgent1State


class Agent1GraphState(TypedDict, total=False):
    korean_sentence: str


def _generate_sentence_node(_: Agent1GraphState) -> Agent1GraphState:
    return {"korean_sentence": generate_korean_sentence()}


def build_agent_1_sub_graph():
    graph_builder = StateGraph(Agent1GraphState)
    graph_builder.add_node("generate_sentence", _generate_sentence_node)
    graph_builder.add_edge(START, "generate_sentence")
    graph_builder.add_edge("generate_sentence", END)
    return graph_builder.compile()


sample_agent_1_sub_graph = build_agent_1_sub_graph()


def run_agent_1() -> SampleAgent1State:
    result = sample_agent_1_sub_graph.invoke({})
    return SampleAgent1State(korean_sentence=result["korean_sentence"])
