"""테스트 케이스 생성 노드: [질병 x 약관] 조합으로 EvaluationTestCase 리스트 생성."""

from app.agents.vet_agent.state import DiseaseInfo, VetAgentState

from app.evaluation.state import EvaluationTestCase


def build_test_cases(
    file_name: str,
    state: VetAgentState,
    diseases: list[DiseaseInfo],
    policy_texts: list[str],
) -> list[EvaluationTestCase]:
    """[질병 x 약관]의 모든 조합에 대해 EvaluationTestCase 리스트를 생성합니다.

    예: 질병 3개 x 약관 3개 = 9개의 테스트 케이스
    """
    hc = state.health_condition
    disease_history = (hc.disease_surgery_history if hc else None) or "없음"

    test_cases: list[EvaluationTestCase] = []

    for disease in diseases:
        for policy_text in policy_texts:
            test_cases.append(
                EvaluationTestCase(
                    file_name=file_name,
                    species=state.species or "미상",
                    breed=state.breed or "미상",
                    age=state.age or 0,
                    disease_surgery_history=disease_history,
                    disease_name=disease.name,
                    policy_text=policy_text,
                )
            )

    return test_cases
