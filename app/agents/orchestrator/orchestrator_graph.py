from pathlib import Path

from langgraph.graph import END, START, StateGraph
from langgraph.checkpoint.memory import InMemorySaver

from rich import print as rprint

from app.agents.utils import create_graph_image

from app.agents.orchestrator.state.orchestrator_state import OrchestratorState
from app.agents.orchestrator.nodes import route_after_user_input
from app.agents.user_input_template_agent.utils.cli import (
    create_arg_parser,
    load_state_from_yaml,
)

from app.agents.rag_agent.rag_graph import graph as retrieve_graph
from app.agents.user_input_template_agent.graph import graph as user_input_graph
from app.agents.vet_agent.graph import graph as vet_graph
from app.agents.judge_agent.graph import graph as judge_graph
from app.agents.composer_agent.graph import graph as composer_graph


def build_orchestrator_graph(checkpointer=None):
    graph_builder = StateGraph(OrchestratorState)

    graph_builder.add_node("user_input_template", user_input_graph)
    graph_builder.add_node("vet_diagnosis", vet_graph)
    graph_builder.add_node("RAG", retrieve_graph)
    graph_builder.add_node("judge", judge_graph)
    graph_builder.add_node("composer", composer_graph)

    graph_builder.add_edge(START, "user_input_template")
    graph_builder.add_conditional_edges("user_input_template", route_after_user_input)
    graph_builder.add_edge("vet_diagnosis", "RAG")
    graph_builder.add_edge("RAG", "judge")
    graph_builder.add_edge("judge", "composer")
    graph_builder.add_edge("composer", END)

    return graph_builder.compile(checkpointer=checkpointer)


in_memory_saver = InMemorySaver()
graph = build_orchestrator_graph(checkpointer=in_memory_saver)
create_graph_image(
    graph,
    file_name="orchestrator_graph",
    base_dir=Path(__file__).resolve().parent,
)


def run_test_orchestration(yaml_path: str | Path, config: dict) -> dict:
    """
    주어진 YAML 파일 경로에서 사용자의 입력 상태를 불러와
    Orchestrator 그래프 파이프라인을 실행하는 테스트 함수입니다.

    Args:
        yaml_path (str | Path): 사용자 입력 데이터가 저장된 YAML 파일 경로
        config (dict): LangGraph config (thread_id 등 configurable 포함)
            >>> config = {"configurable": {"thread_id": "test_user"}}

    Returns:
        dict: 전체 그래프 실행 후 OrchestratorState 결과를 반환합니다.
    """
    state = load_state_from_yaml(yaml_path, OrchestratorState)
    result = graph.invoke(state, config=config)
    rprint(f"질병 목록: {result['diseases']}")
    rprint(
        "RAG 결과: ",
        [doc.page_content for doc in result["retrieved_documents"]],
    )
    rprint("Judge 검증 결과:", result.get("validation_result"))
    rprint("최종 유저 답변:", result.get("final_message"))

    return result


def test_router(config: dict) -> None:
    """라우터 분기(diseases 유무)를 검증하기 위한 테스트 함수."""
    yaml_paths = [
        "app/agents/user_input_template_agent/samples/user_input_insurer_1.yaml",
        "app/agents/user_input_template_agent/samples/user_input_insurer_2.yaml",
    ]
    for yaml_path in yaml_paths:
        run_test_orchestration(yaml_path=yaml_path, config=config)


def main(config: dict) -> None:
    args = create_arg_parser().parse_args()
    run_test_orchestration(args.input, config=config)


if __name__ == "__main__":
    config = {"configurable": {"thread_id": "test_user"}}

    main(config)


# uv run python -m app.agents.orchestrator.orchestrator_graph
# uv run python -m app.agents.orchestrator.orchestrator_graph --input app/agents/user_input_template_agent/samples/user_input_all.yaml
