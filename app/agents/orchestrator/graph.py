from langgraph.graph import END, START, StateGraph
from pydantic import BaseModel

from app.agents.sample_agent_1.graph import sample_agent_1_sub_graph
from app.agents.sample_agent_2.graph import sample_agent_2_sub_graph
from app.agents.rag_agent.retrieve_graph import build_graph
from app.agents.vet_agent.mocks.vet_agent_mock import (
    VetAgentMockState,
    create_mock_vet_agent_state,
)

from rich import print as rprint


class OrchestratorState(BaseModel):
    korean_sentence: str | None = None
    english_sentence: str | None = None
    vet_agent_mock: VetAgentMockState | None = None


def build_orchestrator_graph():
    graph_builder = StateGraph(OrchestratorState)
    retrieve_graph = build_graph()

    graph_builder.add_node("retrieve_graph", retrieve_graph)
    graph_builder.add_node("sample_agent_1_sub_graph", sample_agent_1_sub_graph)
    graph_builder.add_node("sample_agent_2_sub_graph", sample_agent_2_sub_graph)

    graph_builder.add_edge(START, "retrieve_graph")
    graph_builder.add_edge("retrieve_graph", "sample_agent_1_sub_graph")
    graph_builder.add_edge("sample_agent_1_sub_graph", "sample_agent_2_sub_graph")
    graph_builder.add_edge("sample_agent_2_sub_graph", END)

    return graph_builder.compile()


orchestrator_graph = build_orchestrator_graph()


def run_test_orchestration() -> str:
    test_state = OrchestratorState()
    test_state.vet_agent_mock = create_mock_vet_agent_state()
    rprint(">>> test_state", test_state)

    result = orchestrator_graph.invoke(test_state)
    return (
        f"최초 생성된 한글 문장은 {result['korean_sentence']}였으며, "
        f"번역된 결과는 {result['english_sentence']}입니다"
    )


if __name__ == "__main__":
    run_test_orchestration()
