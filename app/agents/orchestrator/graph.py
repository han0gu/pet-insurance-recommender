from typing import TypedDict

from langgraph.graph import END, START, StateGraph

from app.agents.sample_agent_1.graph import sample_agent_1_sub_graph
from app.agents.sample_agent_2.graph import sample_agent_2_sub_graph


class OrchestratorState(TypedDict, total=False):
    korean_sentence: str
    english_sentence: str


def build_orchestrator_graph():
    graph_builder = StateGraph(OrchestratorState)
    graph_builder.add_node("sample_agent_1_sub_graph", sample_agent_1_sub_graph)
    graph_builder.add_node("sample_agent_2_sub_graph", sample_agent_2_sub_graph)
    graph_builder.add_edge(START, "sample_agent_1_sub_graph")
    graph_builder.add_edge("sample_agent_1_sub_graph", "sample_agent_2_sub_graph")
    graph_builder.add_edge("sample_agent_2_sub_graph", END)
    return graph_builder.compile()


orchestrator_graph = build_orchestrator_graph()


def run_test_orchestration() -> str:
    result = orchestrator_graph.invoke({})
    return (
        f"최초 생성된 한글 문장은 {result['korean_sentence']}였으며, "
        f"번역된 결과는 {result['english_sentence']}입니다"
    )
