"""
평가 파이프라인 흐름 정의.

YAML 로드 → 질병 추출(Vet/Mock) → 약관 검색(Mock) → 테스트 케이스 생성
→ Judge + Evaluator 판단 → 라벨 부여. (LangGraph 미사용, 순차 호출)
"""

import logging

from dotenv import load_dotenv
from rich import print as rprint

from app.agents.vet_agent.state import DiseaseInfo, VetAgentState
from app.evaluation.mocks.mock_data import get_mock_diseases, get_mock_policies
from app.evaluation.nodes import (
    build_test_cases,
    compute_label,
    evaluate_test_case,
    judge_predict,
    load_all_yaml_states,
)
from app.evaluation.state import EvaluationRecord, EvaluationTestCase

load_dotenv()
logger = logging.getLogger(__name__)

# True: 실제 Vet Agent 호출 / False: Mock 질병 사용
USE_REAL_VET_AGENT = True
# 로드할 YAML 파일 개수 (None이면 전체)
YAML_LOAD_LIMIT = 1


async def run_vet_agent(state: VetAgentState) -> list[DiseaseInfo]:
    """실제 Vet Agent 그래프를 실행하여 질병 목록을 생성합니다."""
    from app.agents.vet_agent.graph import graph as vet_graph

    input_data = state.model_dump(
        include={
            "species",
            "breed",
            "age",
            "gender",
            "is_neutered",
            "weight",
            "health_condition",
            "coverage_style",
            "preferred_insurers",
        },
        exclude_none=True,
    )

    result = await vet_graph.ainvoke(input_data)
    return [DiseaseInfo.model_validate(d) for d in result.get("diseases", [])]


async def _evaluate_test_cases(
    test_cases: list[EvaluationTestCase],
) -> list[EvaluationRecord]:
    """테스트 케이스 목록에 대해 Judge + Evaluator 판단을 수행하고 EvaluationRecord 리스트를 반환합니다.

    각 테스트 케이스마다 Judge(예측)와 Evaluator(정답)를 호출한 뒤, 비교하여 TP/TN/FP/FN 라벨을 부여합니다.
    """
    records: list[EvaluationRecord] = []
    label_colors = {"TP": "green", "TN": "blue", "FP": "bold red", "FN": "yellow"}

    for tc_idx, test_case in enumerate(test_cases, 1):
        rprint(
            f"    [{tc_idx}/{len(test_cases)}] "
            f"질병='{test_case.disease_name}' 판단 중...",
            end="",
        )

        judge_result = await judge_predict(test_case)
        evaluator_result = await evaluate_test_case(test_case)
        label = compute_label(judge_result, evaluator_result)

        color = label_colors.get(label, "white")
        rprint(
            f" → Judge={'O' if judge_result.is_covered else 'X'} / "
            f"Evaluator={'O' if evaluator_result.is_covered else 'X'} / "
            f"[{color}]{label}[/{color}]"
        )

        records.append(
            EvaluationRecord(
                test_case=test_case,
                judge_prediction=judge_result,
                evaluator_ground_truth=evaluator_result,
                label=label,
            )
        )

    return records


async def run_evaluation_pipeline() -> list[EvaluationRecord]:
    """LLM-as-a-Judge 평가 파이프라인 전체를 실행합니다."""
    rprint("\n[bold cyan]═══ LLM-as-a-Judge 평가 파이프라인 시작 ═══[/bold cyan]\n")

    rprint(f"[1/5] YAML 데이터 로드 중... (limit={YAML_LOAD_LIMIT})")
    yaml_states = load_all_yaml_states(limit=YAML_LOAD_LIMIT)
    rprint(f"  → {len(yaml_states)}개 상태 로드 완료\n")

    all_records: list[EvaluationRecord] = []

    # YAML 파일 단위로 순회: 각 파일마다 질병 추출 → 약관 검색 → 테스트 케이스 생성 → Judge/Evaluator 판단
    for idx, (file_name, state) in enumerate(yaml_states, 1):
        rprint(
            f"[bold]── [{idx}/{len(yaml_states)}] {file_name} "
            f"({state.breed}, {state.age}세) ──[/bold]"
        )

        rprint("  [2/5] 질병 목록 생성 중...")
        if USE_REAL_VET_AGENT:
            diseases = await run_vet_agent(state)
        else:
            diseases = get_mock_diseases(state)
        rprint(f"  → 추출된 질병 {len(diseases)}개: {[d.name for d in diseases]}\n")

        rprint("  [3/5] 약관 텍스트 검색 중... (Mock)")
        policy_texts = get_mock_policies()
        rprint(f"  → 약관 {len(policy_texts)}개 로드\n")

        test_cases = build_test_cases(file_name, state, diseases, policy_texts)
        rprint(
            f"  [4/5] 테스트 케이스 생성: {len(diseases)} 질병 × "
            f"{len(policy_texts)} 약관 = {len(test_cases)}개\n"
        )

        rprint("  [5/5] Judge + Evaluator 판단 실행 중...")
        file_records = await _evaluate_test_cases(test_cases)
        all_records.extend(file_records)

        rprint()

    return all_records
