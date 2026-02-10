from langgraph.graph import END, START, StateGraph

from app.agents.user_input_template_agent.graph import graph as user_input_graph
from app.agents.vet_agent.state import VetAgentState
from app.agents.vet_agent.graph import graph as vet_graph
from app.agents.rag_agent.retrieve_graph import build_graph, RagState


class OrchestratorState(VetAgentState, RagState): ...


def build_orchestrator_graph():
    graph_builder = StateGraph(OrchestratorState)
    retrieve_graph = build_graph()

    graph_builder.add_node("user_input_template", user_input_graph)
    graph_builder.add_node("vet_diagnosis", vet_graph)
    graph_builder.add_node("retrieve_graph", retrieve_graph)

    graph_builder.add_edge(START, "user_input_template")
    graph_builder.add_edge("user_input_template", "vet_diagnosis")
    graph_builder.add_edge("vet_diagnosis", "retrieve_graph")
    graph_builder.add_edge("retrieve_graph", END)

    return graph_builder.compile()


orchestrator_graph = build_orchestrator_graph()


def run_test_orchestration() -> str:
    result = orchestrator_graph.invoke(
        OrchestratorState(
            species="개",
            breed="골든 리트리버",
            age=5,
            gender="male",
            weight=30,
        )
    )
    return f"질병 목록: {result['diseases']}"


if __name__ == "__main__":
    from rich import print as rprint

    result = run_test_orchestration()
    rprint(result)
