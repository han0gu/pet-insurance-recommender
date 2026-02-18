"""
정답지 생성 LLM (Evaluator) 노드.

금융감독원 수준의 매우 엄격한 '보험 심사 평가자' 프롬프트를 사용하여
각 [질병 x 약관] 조합에 대한 보장 여부 정답을 생성합니다.
"""

from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate
from langchain_upstage import ChatUpstage

from app.evaluation.state import EvaluationTestCase, EvaluatorGroundTruth

load_dotenv()

EVALUATOR_SYSTEM_PROMPT = """\
당신은 금융감독원 출신의 **보험 약관 심사 최고 전문가**입니다.
20년 이상의 보험 심사 경력을 가지고 있으며, 약관 문구를 **글자 그대로** 해석합니다.

# 당신의 역할
주어진 [질병명]과 [약관 텍스트]를 분석하여, 해당 질병이 이 약관에 의해 보장되는지 판단합니다.

# 판단 기준 (최상위 우선순위 순서)

1. **면책 조항 확인 (최우선)**
   - 약관의 '보상하지 않는 손해', '면책', '보장 제외' 항목에 해당 질병이
     **명시적으로 언급**되어 있으면 → is_covered = False
   - 질병명이 완전히 동일하지 않더라도, 의학적으로 동일한 질병이면 면책 판정

2. **기왕증(기존 질병) 확인**
   - 반려동물의 '기저질환/수술 이력'에 해당 질병 또는 관련 질병이 이미 존재하고,
     약관에 기왕증 면책 조항이 있으면 → is_covered = False

3. **보장 범위 확인**
   - 약관에 해당 질병이 보장 항목(입원, 수술, 통원 등)에 포함되면 → is_covered = True
   - 직접 언급은 없지만 상위 카테고리로 보장 가능하면 → is_covered = True

4. **불확실한 경우의 원칙 (보수적 판단)**
   - 약관 문구만으로 명확히 판단 불가 시 → is_covered = False
   - reason에 왜 불확실한지 구체적으로 서술

# 출력 규칙
- is_covered: True(보장됨) 또는 False(보장 안 됨 / 면책)
- reason: 약관의 **구체적인 문구**를 인용하며, 어떤 조항에 근거했는지 2~3문장으로 서술
"""

EVALUATOR_HUMAN_PROMPT = """\
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


async def evaluate_test_case(
    test_case: EvaluationTestCase,
) -> EvaluatorGroundTruth:
    """평가용 LLM(solar-pro2)으로 [질병 x 약관] 보장 여부의 '정답'을 생성합니다."""
    llm = ChatUpstage(model="solar-pro2", temperature=0)
    structured_llm = llm.with_structured_output(EvaluatorGroundTruth)

    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", EVALUATOR_SYSTEM_PROMPT),
            ("human", EVALUATOR_HUMAN_PROMPT),
        ]
    )

    chain = prompt | structured_llm
    result: EvaluatorGroundTruth = await chain.ainvoke(
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
