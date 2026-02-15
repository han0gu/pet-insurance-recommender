from langgraph.graph.state import CompiledStateGraph
from langgraph.graph import StateGraph, START, END

from app.agents import utils

from app.agents.rag_agent.nodes.embed_query import embed_query
from app.agents.rag_agent.nodes.generate_user_query import generate_user_query
from app.agents.rag_agent.nodes.retrieve import retrieve_normal, retrieve_simple
from app.agents.rag_agent.nodes.summary import summary
from app.agents.rag_agent.state.rag_state import RagState

from app.agents.vet_agent.state import VetAgentState


class RagGraphState(VetAgentState, RagState): ...


def build_graph() -> CompiledStateGraph:
    workflow = StateGraph(
        RagGraphState,
        # input_schema=VetAgentState,
        # output_schema=RagState,
    )

    workflow.add_node("generate_user_query", generate_user_query)
    workflow.add_node("embed_query", embed_query)
    workflow.add_node("retrieve_normal", retrieve_normal)
    workflow.add_node("retrieve_simple", retrieve_simple)
    workflow.add_node("summary", summary)

    workflow.add_edge(START, "generate_user_query")
    workflow.add_edge("generate_user_query", "embed_query")
    workflow.add_edge("embed_query", "retrieve_normal")
    workflow.add_edge("embed_query", "retrieve_simple")
    workflow.add_edge("retrieve_normal", "summary")
    workflow.add_edge("retrieve_simple", "summary")
    workflow.add_edge("summary", END)

    return workflow.compile()


graph = build_graph()

if __name__ == "__main__":
    retrieve_graph = build_graph()

    utils.create_graph_image(
        retrieve_graph,
        utils.get_current_file_name(__file__, True),
        utils.get_parent_path(__file__),
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


# uv run python -m app.agents.rag_agent.rag_graph
