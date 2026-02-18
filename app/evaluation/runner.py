"""
LLM-as-a-Judge 평가 파이프라인 실행 로직.

전체 흐름:
  1) data_loader로 YAML 상태를 불러옴 (기본 3개 슬라이싱)
  2) Vet Agent를 통과시켜 질병 목록 생성 (또는 Mock 함수 사용)
  3) RAG 검색 결과 가져옴 (기본: Mock 약관 3개)
  4) [1질병 x 1약관] 조합으로 EvaluationTestCase 생성
  5) JudgeAgent 로직 + Evaluator LLM에 각각 통과시켜 예측/정답 비교

실행 방법:
    uv run python -m app.evaluation.runner
"""

import asyncio
import logging

from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate
from langchain_upstage import ChatUpstage
from rich import print as rprint

from app.agents.vet_agent.state import DiseaseInfo, VetAgentState
from app.evaluation.data_loader import load_all_yaml_states
from app.evaluation.evaluator import evaluate_test_case
from app.evaluation.metrics import compute_and_display_metrics, save_results_to_csv
from app.evaluation.schemas import (
    EvaluationRecord,
    EvaluationTestCase,
    EvaluatorGroundTruth,
    JudgePrediction,
)

load_dotenv()
logger = logging.getLogger(__name__)


# ==========================================
# 1. Mock 데이터: 약관 텍스트 (기본 연결)
# ==========================================

MOCK_POLICY_TEXTS: list[str] = [
    """[상품명: 메리츠 펫보험 프리미엄]
제4조 (보장하는 손해)
① 이 특별약관에서는 보험기간 중 피보험자(반려동물)에게 발생한 상해 또는
   질병으로 인하여 동물병원에서 수의사의 치료를 받은 경우 의료비를 보상합니다.
② 보장 범위: 입원, 통원(외래), 수술
③ 보상 비율: 실제 발생 의료비의 70%

제5조 (보상하지 않는 손해)
① 선천성 질환 및 유전성 질환
② 가입 전 이미 진단된 질병이나 증상 (기왕증)
③ 치과 질환 (치석 제거, 발치 등)
④ 미용 목적의 시술
⑤ 예방 접종 및 건강검진 비용""",
    """[상품명: KB 펫보험 든든보장]
제3조 (보장하는 손해)
① 이 보험에서는 피보험자(반려동물)의 질병 또는 상해로 인한 동물의료비를 보장합니다.
② 입원의료비: 1일당 15만원 한도
③ 수술비: 1회당 100만원 한도
④ 통원의료비: 1일당 5만원 한도, 연간 30회 한도

제6조 (보상하지 않는 손해)
① 보험 개시일로부터 30일 이내에 발생한 질병
② 슬개골 탈구 1기 ~ 2기 (보험 가입 시 이미 진단된 경우)
③ 임신·출산·유산 관련 질환
④ 보험사기 의심 건""",
    """[상품명: 삼성화재 펫보험 안심플랜]
제2조 (보장 범위)
① 질병 입원: 연간 500만원 한도
② 질병 통원: 1일 5만원, 연간 100만원 한도
③ 질병 수술: 1회 200만원, 연간 400만원 한도
④ 피부질환 특약: 연간 50만원 한도

제7조 (보상하지 않는 손해)
① 기왕증 (가입 전 이미 진단된 질병)
② 선천성 질환
③ 한방 치료 및 대체의학 비용
④ 심장사상충 예방 미실시로 인한 심장사상충 감염""",
]


def get_mock_policies() -> list[str]:
    """Mock 약관 텍스트 3개를 반환합니다.

    실제 RAG 파이프라인 연동 전까지 사용하는 대체 함수입니다.
    실제 연동 시 이 함수를 RAG Agent 호출로 교체하세요.
    """
    return MOCK_POLICY_TEXTS


# ==========================================
# 2. Mock 데이터: Vet Agent 질병 추출 (옵션)
# ==========================================


def get_mock_diseases(state: VetAgentState) -> list[DiseaseInfo]:
    """Vet Agent 대신 사용할 Mock 질병 목록을 반환합니다.

    API 비용 절감이 필요할 때 사용합니다.
    disease_surgery_history(과거 병력)도 질병으로 포함시킵니다.
    """
    mock_diseases = [
        DiseaseInfo(name="슬개골 탈구", incidence_rate="높음", onset_period="전 연령"),
        DiseaseInfo(
            name="치과 질환(치주염 등)", incidence_rate="중간", onset_period="5세 이상"
        ),
    ]

    # 과거 병력이 있으면 질병 목록에 추가
    hc = state.health_condition
    if hc and hc.disease_surgery_history:
        mock_diseases.append(
            DiseaseInfo(
                name=hc.disease_surgery_history,
                incidence_rate="기왕증",
                onset_period="가입 전",
            )
        )

    return mock_diseases


# ==========================================
# 3. 실제 Vet Agent 연동 함수
# ==========================================


async def run_vet_agent(state: VetAgentState) -> list[DiseaseInfo]:
    """실제 Vet Agent 그래프를 실행하여 질병 목록을 생성합니다.

    Vet Agent는 반려동물 정보(종, 품종, 나이, 건강상태 등)를 기반으로
    해당 동물이 잘 걸리는 질병을 LLM으로 분석합니다.

    Args:
        state: YAML에서 로드한 반려동물 상태 정보

    Returns:
        DiseaseInfo 리스트 (Vet Agent가 추출한 질병 목록)
    """
    from app.agents.vet_agent.graph import graph as vet_graph

    # UserInputTemplateState 필드만 추출하여 입력으로 전달
    # (Vet Agent의 input_schema가 UserInputTemplateState이므로)
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


# ==========================================
# 4. JudgeAgent 단건 판단 함수
# ==========================================

# 기존 validator_node의 프롬프트를 단건 [1질병 x 1약관] 판단용으로 변형
JUDGE_SYSTEM_PROMPT = """\
당신은 보험 약관 심사 전문가입니다.
주어진 반려동물 정보와 질병명, 약관 텍스트를 비교하여 해당 질병의 보장 여부를 판단하세요.

# 심사 기준 (엄격 준수)
1. **질병 면책 (최우선 순위)**
   - 해당 질병이 약관의 '보상하지 않는 손해(면책 사항)'에 포함되는지 확인

2. **기왕증(기존 질병) 확인**
   - 기저질환/수술 이력에 해당 질병이 이미 있고,
     약관에 기왕증 면책 조항이 있으면 보장 불가

3. **나이 제한**
   - 반려동물의 나이가 약관의 가입/갱신 연령 제한을 초과하는지 확인

4. **보장 범위 확인**
   - 약관의 보장 항목(입원, 수술, 통원 등)에 해당 질병이 포함되는지 확인
"""

JUDGE_HUMAN_PROMPT = """\
다음 정보를 바탕으로 보장 여부를 판단해 주세요.

=== [반려동물 정보] ===
- 종: {species}
- 품종: {breed}
- 나이: {age}세
- 기저질환/수술 이력: {disease_surgery_history}

=== [평가 대상 질병] ===
{disease_name}

=== [약관 텍스트] ===
{policy_text}
"""


async def judge_predict(test_case: EvaluationTestCase) -> JudgePrediction:
    """기존 JudgeAgent 로직을 단건 [1질병 x 1약관]에 대해 실행합니다.

    기존 validator_node는 다수의 질병과 약관을 한꺼번에 평가하지만,
    이 함수는 Confusion Matrix 계산을 위해 단건 단위로 동작합니다.

    Args:
        test_case: 유저 상태 + 질병 1개 + 약관 1개로 구성된 테스트 케이스

    Returns:
        JudgePrediction: JudgeAgent의 판단 결과 (is_covered, reason)
    """
    llm = ChatUpstage(model="solar-pro2", temperature=0)
    structured_llm = llm.with_structured_output(JudgePrediction)

    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", JUDGE_SYSTEM_PROMPT),
            ("human", JUDGE_HUMAN_PROMPT),
        ]
    )

    chain = prompt | structured_llm
    result: JudgePrediction = await chain.ainvoke(
        {
            "species": test_case.species,
            "breed": test_case.breed,
            "age": test_case.age,
            "disease_surgery_history": test_case.disease_surgery_history or "없음",
            "disease_name": test_case.disease_name,
            "policy_text": test_case.policy_text,
        }
    )

    return result


# ==========================================
# 5. 혼동 행렬 라벨 계산
# ==========================================


def compute_label(judge: JudgePrediction, evaluator: EvaluatorGroundTruth) -> str:
    """Judge 예측값과 Evaluator 정답값을 비교하여 TP/TN/FP/FN을 반환합니다.

    Positive = 보장됨(is_covered=True), Negative = 보장 안 됨(is_covered=False)
    - TP: 둘 다 보장된다고 판단 (정확한 보장 판정)
    - TN: 둘 다 보장 안 된다고 판단 (정확한 면책 판정)
    - FP: Judge만 보장된다고 판단 (위험! 실제로는 보장 안 됨)
    - FN: Judge만 보장 안 된다고 판단 (보수적 판단, 실제로는 보장됨)
    """
    if judge.is_covered and evaluator.is_covered:
        return "TP"
    if not judge.is_covered and not evaluator.is_covered:
        return "TN"
    if judge.is_covered and not evaluator.is_covered:
        return "FP"
    return "FN"


# ==========================================
# 6. 테스트 케이스 생성 헬퍼
# ==========================================


def build_test_cases(
    file_name: str,
    state: VetAgentState,
    diseases: list[DiseaseInfo],
    policy_texts: list[str],
) -> list[EvaluationTestCase]:
    """[질병 x 약관]의 모든 조합에 대해 EvaluationTestCase 리스트를 생성합니다.

    예: 질병 3개 x 약관 3개 = 9개의 테스트 케이스
    """
    # health_condition에서 기저질환 정보를 안전하게 추출
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


# ==========================================
# 7. 메인 파이프라인
# ==========================================

# True: 실제 Vet Agent 호출 / False: Mock 질병 사용
USE_REAL_VET_AGENT = True

# 로드할 YAML 파일 개수 (API 비용 절감용, None이면 전체)
YAML_LOAD_LIMIT = 3


async def run_evaluation_pipeline() -> list[EvaluationRecord]:
    """LLM-as-a-Judge 평가 파이프라인 전체를 실행합니다.

    Returns:
        EvaluationRecord 리스트 (모든 테스트 케이스의 결과)
    """
    rprint("\n[bold cyan]═══ LLM-as-a-Judge 평가 파이프라인 시작 ═══[/bold cyan]\n")

    # ── Step 1: YAML 상태 로드 ──
    rprint(f"[1/5] YAML 데이터 로드 중... (limit={YAML_LOAD_LIMIT})")
    yaml_states = load_all_yaml_states(limit=YAML_LOAD_LIMIT)
    rprint(f"  → {len(yaml_states)}개 상태 로드 완료\n")

    all_records: list[EvaluationRecord] = []

    for idx, (file_name, state) in enumerate(yaml_states, 1):
        rprint(
            f"[bold]── [{idx}/{len(yaml_states)}] {file_name} "
            f"({state.breed}, {state.age}세) ──[/bold]"
        )

        # ── Step 2: 질병 목록 생성 ──
        rprint("  [2/5] 질병 목록 생성 중...")
        if USE_REAL_VET_AGENT:
            # 실제 Vet Agent 사용
            diseases = await run_vet_agent(state)
        else:
            # Mock 질병 사용 (API 비용 절감)
            diseases = get_mock_diseases(state)

        rprint(f"  → 추출된 질병 {len(diseases)}개: {[d.name for d in diseases]}\n")

        # ── Step 3: RAG 약관 검색 (Mock) ──
        rprint("  [3/5] 약관 텍스트 검색 중... (Mock)")
        policy_texts = get_mock_policies()
        rprint(f"  → 약관 {len(policy_texts)}개 로드\n")

        # ── Step 4: 테스트 케이스 생성 ──
        test_cases = build_test_cases(file_name, state, diseases, policy_texts)
        rprint(
            f"  [4/5] 테스트 케이스 생성: {len(diseases)} 질병 × "
            f"{len(policy_texts)} 약관 = {len(test_cases)}개\n"
        )

        # ── Step 5: Judge + Evaluator 실행 ──
        rprint("  [5/5] Judge + Evaluator 판단 실행 중...")
        for tc_idx, test_case in enumerate(test_cases, 1):
            rprint(
                f"    [{tc_idx}/{len(test_cases)}] "
                f"질병='{test_case.disease_name}' 판단 중...",
                end="",
            )

            # Judge와 Evaluator를 순차 실행 (Rate Limit 방지)
            judge_result = await judge_predict(test_case)
            evaluator_result = await evaluate_test_case(test_case)

            label = compute_label(judge_result, evaluator_result)

            # 라벨 색상: FP(위험)은 빨간색, TP는 초록색
            label_color = {
                "TP": "green",
                "TN": "blue",
                "FP": "bold red",
                "FN": "yellow",
            }.get(label, "white")

            rprint(
                f" → Judge={'O' if judge_result.is_covered else 'X'} / "
                f"Evaluator={'O' if evaluator_result.is_covered else 'X'} / "
                f"[{label_color}]{label}[/{label_color}]"
            )

            all_records.append(
                EvaluationRecord(
                    test_case=test_case,
                    judge_prediction=judge_result,
                    evaluator_ground_truth=evaluator_result,
                    label=label,
                )
            )

        rprint()

    return all_records


# ==========================================
# 8. 엔트리포인트
# ==========================================


async def main() -> None:
    """평가 파이프라인을 실행하고 결과를 출력/저장합니다."""
    records = await run_evaluation_pipeline()

    if not records:
        rprint("[bold red]평가 결과가 없습니다. YAML 데이터를 확인하세요.[/bold red]")
        return

    # 혼동 행렬 출력 + CSV 저장
    compute_and_display_metrics(records)
    save_results_to_csv(records)


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    asyncio.run(main())
