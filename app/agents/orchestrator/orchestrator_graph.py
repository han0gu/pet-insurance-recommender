from pathlib import Path

from langgraph.graph import END, START, StateGraph

from app.agents.utils import create_graph_image

from app.agents.user_input_template_agent.graph import graph as user_input_graph
from app.agents.vet_agent.state import VetAgentState
from app.agents.vet_agent.graph import graph as vet_graph
from app.agents.rag_agent.rag_graph import build_graph, RagState


class OrchestratorState(VetAgentState, RagState): ...


def build_orchestrator_graph():
    graph_builder = StateGraph(OrchestratorState)
    retrieve_graph = build_graph()

    graph_builder.add_node("user_input_template", user_input_graph)
    graph_builder.add_node("vet_diagnosis", vet_graph)
    graph_builder.add_node("RAG", retrieve_graph)

    graph_builder.add_edge(START, "user_input_template")
    graph_builder.add_edge("user_input_template", "vet_diagnosis")
    graph_builder.add_edge("vet_diagnosis", "RAG")
    graph_builder.add_edge("RAG", END)

    return graph_builder.compile()


orchestrator_graph = build_orchestrator_graph()
create_graph_image(
    orchestrator_graph,
    file_name="orchestrator_graph",
    base_dir=Path(__file__).resolve().parent,
)


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
