"""JudgeAgent 단건 판단 노드: [1질병 x 1약관]에 대해 보장 여부 판단 및 라벨 계산."""

from langchain_core.prompts import ChatPromptTemplate
from langchain_upstage import ChatUpstage

from app.evaluation.state import (
    EvaluationTestCase,
    EvaluatorGroundTruth,
    JudgePrediction,
)

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
    """기존 JudgeAgent 로직을 단건 [1질병 x 1약관]에 대해 실행합니다."""
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


def compute_label(judge: JudgePrediction, evaluator: EvaluatorGroundTruth) -> str:
    """Judge 예측값과 Evaluator 정답값을 비교하여 TP/TN/FP/FN을 반환합니다.

    Positive = 보장됨(is_covered=True), Negative = 보장 안 됨(is_covered=False)
    """
    if judge.is_covered and evaluator.is_covered:
        return "TP"
    if not judge.is_covered and not evaluator.is_covered:
        return "TN"
    if judge.is_covered and not evaluator.is_covered:
        return "FP"
    return "FN"
