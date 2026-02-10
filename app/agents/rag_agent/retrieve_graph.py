import io
from pathlib import Path

from PIL import Image as PILImage

from langgraph.graph import StateGraph, START, END

from rich import print as rprint

from app.agents.rag_agent.nodes.embed_query import embed_query
from app.agents.rag_agent.nodes.generate_user_query import generate_user_query
from app.agents.rag_agent.state.rag_state import RagState
from app.agents.rag_agent.tools.retrieve import retrieve

from app.agents.vet_agent.state import VetAgentState


def build_graph():
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
    app = build_graph()

    png_data = app.get_graph().draw_mermaid_png()

    # PIL을 사용하여 콘솔/로컬 환경에서 이미지 띄우기
    # 바이너리 데이터를 메모리 상의 이미지로 변환합니다.
    img = PILImage.open(io.BytesIO(png_data))

    # 파일로 저장
    BASE_DIR = Path(__file__).resolve().parent  # app/agents/rag_agent/
    img.save(f"{BASE_DIR}/retrieve_graph.png")

    # 시스템 기본 이미지 뷰어로 이미지를 엽니다.
    # img.show()

    #
    result = app.invoke(
        VetAgentState(
            species="강아지",
            breed="치와와",
            age=10,
            gender="male",
            weight=10,
        )
    )
    # rprint(">>> retrieve result", result["retrieved_documents"])

# uv run python -m app.agents.rag_agent.retrieve_graph
