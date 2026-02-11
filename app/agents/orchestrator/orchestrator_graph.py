from pathlib import Path

from langgraph.graph import END, START, StateGraph

from rich import print as rprint

from app.agents.utils import create_graph_image

from app.agents.orchestrator.state.orchestrator_state import OrchestratorState

from app.agents.rag_agent.rag_graph import graph as retrieve_graph
from app.agents.user_input_template_agent.graph import graph as user_input_graph
from app.agents.vet_agent.graph import graph as vet_graph
from app.agents.judge_agent.graph import graph as judge_graph
from app.agents.composer_agent.graph import graph as composer_graph



def build_orchestrator_graph():
    graph_builder = StateGraph(OrchestratorState)

    graph_builder.add_node("user_input_template", user_input_graph)
    graph_builder.add_node("vet_diagnosis", vet_graph)
    graph_builder.add_node("RAG", retrieve_graph)
    graph_builder.add_node("judge", judge_graph)
    graph_builder.add_node("composer", composer_graph)

    graph_builder.add_edge(START, "user_input_template")
    graph_builder.add_edge("user_input_template", "vet_diagnosis")
    graph_builder.add_edge("vet_diagnosis", "RAG")
    graph_builder.add_edge("RAG", "judge")
    graph_builder.add_edge("judge", "composer")
    graph_builder.add_edge("composer", END)

    return graph_builder.compile()


graph = build_orchestrator_graph()
create_graph_image(
    graph,
    file_name="orchestrator_graph",
    base_dir=Path(__file__).resolve().parent,
)


def run_test_orchestration() -> str:
    result = graph.invoke(
        OrchestratorState(
            species="개",
            breed="골든 리트리버",
            age=5,
            gender="male",
            weight=30,
        )
    )
    rprint(f"질병 목록: {result['diseases']}")
    rprint(
        "RAG 결과: ",
        [doc.page_content for doc in result["retrieved_documents"]],
    )
    rprint("Judge 검증 결과:", result.get("validation_result"))
    rprint("최종 유저 답변:", result.get("final_message"))

    return result


if __name__ == "__main__":
    run_test_orchestration()


# uv run python -m app.agents.orchestrator.orchestrator_graph
