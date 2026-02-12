from pathlib import Path

from langgraph.graph import END, START, StateGraph

from rich import print as rprint

from app.agents.utils import create_graph_image

from app.agents.orchestrator.state.orchestrator_state import OrchestratorState
from app.agents.user_input_template_agent.utils.cli import (
    create_arg_parser,
    load_state_from_yaml,
)

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


def run_test_orchestration(yaml_path: str | Path) -> str:
    """
    주어진 YAML 파일 경로에서 사용자의 입력 상태를 불러와 Orchestrator 그래프 파이프라인을 실행하는 테스트 함수입니다.

    Args:
        yaml_path (str | Path): 사용자 입력 데이터가 저장된 YAML 파일 경로

    Returns:
        dict: 전체 그래프 실행 후 OrchestratorState 결과를 반환합니다.

    실행 예시:
        >>> run_test_orchestration("app/agents/user_input_template_agent/samples/user_input_all.yaml")
        질병 목록: ['슬개골 탈구', '알레르기 피부염']
        RAG 결과:  ['슬개골 탈구는 소형견에서 흔히 나타나는 질환으로 ...', '알레르기 피부염에 관한 최신 동향 ...']
        Judge 검증 결과: {'is_valid': True, 'reason': '수의사 진단 및 RAG 정보 일치'}
        최종 유저 답변: "당신의 반려견(치와와, 10세, 남, 중성화)이 경험할 수 있는 주요 질병은 슬개골 탈구, 알레르기 피부염입니다. 추천 보장 스타일: 최소. 선호 보험사: 메리츠화재해상보험, 삼성화재해상보험, DB손해보험."

    """
    state = load_state_from_yaml(yaml_path, OrchestratorState)
    result = graph.invoke(state)
    rprint(f"질병 목록: {result['diseases']}")
    rprint(
        "RAG 결과: ",
        [doc.page_content for doc in result["retrieved_documents"]],
    )
    rprint("Judge 검증 결과:", result.get("validation_result"))
    rprint("최종 유저 답변:", result.get("final_message"))

    return result


def main():
    args = create_arg_parser().parse_args()
    run_test_orchestration(args.input)


if __name__ == "__main__":
    main()


# uv run python -m app.agents.orchestrator.orchestrator_graph
# uv run python -m app.agents.orchestrator.orchestrator_graph --input app/agents/user_input_template_agent/samples/user_input_all.yaml
