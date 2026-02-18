"""Mock 약관 텍스트 및 Mock 질병 목록 (API 비용 절감용)."""

from app.agents.vet_agent.state import DiseaseInfo, VetAgentState

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
    """Mock 약관 텍스트 3개를 반환합니다. 실제 RAG 연동 전까지 사용."""
    return MOCK_POLICY_TEXTS


def get_mock_diseases(state: VetAgentState) -> list[DiseaseInfo]:
    """Vet Agent 대신 사용할 Mock 질병 목록. disease_surgery_history도 포함."""
    mock_diseases = [
        DiseaseInfo(name="슬개골 탈구", incidence_rate="높음", onset_period="전 연령"),
        DiseaseInfo(
            name="치과 질환(치주염 등)", incidence_rate="중간", onset_period="5세 이상"
        ),
    ]
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
