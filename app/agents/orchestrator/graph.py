from langgraph.graph import END, START, StateGraph

from user_input_template_agent.state import UserInputTemplateState
from vet_agent.state import VetAgentOutputState
from user_input_template_agent.graph import graph as user_input_graph
from vet_agent.graph import graph as vet_graph


class OrchestratorState(UserInputTemplateState, VetAgentOutputState): ...


def build_orchestrator_graph():
    graph_builder = StateGraph(OrchestratorState)
    graph_builder.add_node("user_input_template", user_input_graph)
    graph_builder.add_node("vet_diagnosis", vet_graph)
    graph_builder.add_edge(START, "user_input_template")
    graph_builder.add_edge("user_input_template", "vet_diagnosis")
    graph_builder.add_edge("vet_diagnosis", END)
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
