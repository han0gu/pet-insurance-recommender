from pathlib import Path

from langgraph.graph.state import CompiledStateGraph
from langgraph.graph import StateGraph, START, END

from rich import print as rprint

from app.agents.utils import create_graph_image

from app.agents.rag_agent.nodes.embed_query import embed_query
from app.agents.rag_agent.nodes.generate_user_query import generate_user_query
from app.agents.rag_agent.state.rag_state import RagState
from app.agents.rag_agent.tools.retrieve import retrieve

from app.agents.vet_agent.state import VetAgentState


def build_graph() -> CompiledStateGraph:
    workflow = StateGraph(
        RagState,
        input_schema=VetAgentState,
        output_schema=RagState,
    )

    workflow.add_node("generate_user_query", generate_user_query)
    workflow.add_node("embed_query", embed_query)
    workflow.add_node("retrieve", retrieve)

    workflow.add_edge(START, "generate_user_query")
    workflow.add_edge("generate_user_query", "embed_query")
    workflow.add_edge("embed_query", "retrieve")
    workflow.add_edge("retrieve", END)

    return workflow.compile()


if __name__ == "__main__":
    retrieve_graph = build_graph()

    create_graph_image(
        retrieve_graph, "retrieve_graph", Path(__file__).resolve().parent
    )

    result = retrieve_graph.invoke(
        VetAgentState(
            species="강아지",
            breed="치와와",
            age=10,
            gender="male",
            weight=10,
        )
    )


# uv run python -m app.agents.rag_agent.retrieve_graph
