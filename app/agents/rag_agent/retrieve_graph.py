import io
from pathlib import Path

from typing import List, Optional
from PIL import Image as PILImage
from pydantic import BaseModel, Field

from langchain_core.documents import Document
from langchain.chat_models import init_chat_model

from langgraph.graph import StateGraph, START, END

from rich import print as rprint

from app.agents.document_parser.nodes.embeddings import load_underlying_embeddings
from app.agents.document_parser.nodes.vector_store import setup_vector_store

from app.agents.vet_agent.state import VetAgentState


BASE_DIR = Path(__file__).resolve().parent  # app/agents/rag_agent/
FILE_NAME = "meritz_terms_normal_1_5.pdf"


class RagState(BaseModel):
    user_query: Optional[str] | None = None
    user_query_embedding: Optional[List[float]] = Field(default_factory=list)
    retrieved_documents: Optional[List[Document]] = Field(default_factory=list)


class Output(BaseModel):
    """학생의 답변을 채점한 결과입니다."""

    user_query: str = Field(
        description="주어진 정보를 바탕으로 LLM이 생성한 사용자 질문"
    )


def generate_user_query(state: VetAgentState) -> RagState:
    rprint(">>> generate_user_query input state", state)

    MODEL = "solar-pro2"
    llm = init_chat_model(model=MODEL, temperature=0.0)
    structured_llm = llm.with_structured_output(Output)
    prompt = f"""
펫 보험에 대해 문의를 남긴 사용자의 정보를 알려줄게. 

종: {state.species}
품종: {state.breed}
나이: {state.age}
성별: {state.gender.name}
체중: {state.weight}

종,품종,나이,성별,체중과 같은 사용자의 정보를 모두 포함해서, 마치 사용자가 직접 질문한 것처럼 자연스러운 질문 문장으로 만들어줘. 생성된 문장에는 사용자의 정보가 하나도 누락되지 않고 포함되어야 해.
"""
    rprint(">>> generated prompt", prompt)

    llm_response: Output = structured_llm.invoke(
        # [{"role": "system", "content": prompt}]
        prompt
    )
    rprint(">>> llm_response", llm_response)

    return {"user_query": llm_response.user_query}


def embed_query(state: RagState) -> RagState | None:
    # rprint("embed_query input state", state)

    if not state.user_query:
        print("user_query error !")
        return

    underlying_embeddings = load_underlying_embeddings()
    user_query_embedding = underlying_embeddings.embed_query(state.user_query)
    return {"user_query_embedding": user_query_embedding}


def retrieve(state: RagState) -> RagState:
    # rprint("retrieve input state", state)

    underlying_embeddings = load_underlying_embeddings()

    vector_store = setup_vector_store(
        underlying_embeddings=underlying_embeddings, collection_name=FILE_NAME
    )

    search_result = vector_store.similarity_search_by_vector(
        state.user_query_embedding, k=3
    )

    return {"retrieved_documents": search_result}


def print_result(state: RagState) -> dict[str, List[Document]]:
    rprint(">>> print_result", state.retrieved_documents)
    return state


def build_graph():
    workflow = StateGraph(
        RagState,
        input_schema=VetAgentState,
        output_schema=RagState,
    )

    workflow.add_node("generate_user_query", generate_user_query)
    workflow.add_node("embed_query", embed_query)
    workflow.add_node("retrieve", retrieve)
    workflow.add_node("print_result", print_result)

    workflow.add_edge(START, "generate_user_query")
    workflow.add_edge("generate_user_query", "embed_query")
    workflow.add_edge("embed_query", "retrieve")
    workflow.add_edge("retrieve", "print_result")
    workflow.add_edge("print_result", END)

    return workflow.compile()


if __name__ == "__main__":
    app = build_graph()

    png_data = app.get_graph().draw_mermaid_png()

    # PIL을 사용하여 콘솔/로컬 환경에서 이미지 띄우기
    # 바이너리 데이터를 메모리 상의 이미지로 변환합니다.
    img = PILImage.open(io.BytesIO(png_data))

    # 파일로 저장
    img.save(f"{BASE_DIR}/retrieve_graph.png")

    # 시스템 기본 이미지 뷰어로 이미지를 엽니다.
    # img.show()

    #
    result = app.invoke(
        RagState(
            species="강아지",
            breed="치와와",
            age=10,
            gender="male",
            weight=10,
        )
    )
