"""반려동물 보험 추천 시스템 - Streamlit 데모 UI"""

import streamlit as st

from app.agents.orchestrator.orchestrator_graph import graph
from app.agents.user_input_template_agent.utils.cli import make_config
from app.agents.user_input_template_agent.state.user_input_template_state import (
    Insurer,
)

# ── 설정 ──────────────────────────────────────────────
INSURER_OPTIONS = [e.value for e in Insurer]
COVERAGE_KR = {"minimal": "최소 보장", "comprehensive": "종합 보장"}

st.set_page_config(page_title="반려동물 보험 추천", layout="wide")
st.title("반려동물 보험 추천 시스템")

# ── 사이드바: 입력 폼 ────────────────────────────────
with st.sidebar:
    st.header("반려동물 정보 입력")

    species = st.selectbox("종", ["강아지", "고양이"])
    breed = st.text_input("품종", placeholder="예: 말티즈, 페르시안")
    age = st.number_input("나이 (세)", min_value=0, max_value=30, value=3)
    gender = st.radio("성별", ["수컷", "암컷"], horizontal=True)
    is_neutered = st.checkbox("중성화 여부")
    weight = st.number_input("체중 (kg)", min_value=1, max_value=100, value=5)

    st.markdown("---")
    st.subheader("보장 선호")
    coverage = st.radio(
        "보장 스타일", ["선택 안함", "최소 보장", "종합 보장"], horizontal=True
    )
    selected_insurers = st.multiselect("선호 보험사", INSURER_OPTIONS)

    st.markdown("---")
    st.subheader("건강 상태 (선택)")
    illness_area = st.text_input("자주 아픈 부위", placeholder="예: 피부, 관절")
    disease_history = st.text_input("질병/수술 이력", placeholder="예: 슬개골 탈구")

    run_btn = st.button("추천 실행", type="primary", use_container_width=True)

# ── state dict 구성 ───────────────────────────────────
gender_map = {"수컷": "male", "암컷": "female"}
coverage_map = {"최소 보장": "minimal", "종합 보장": "comprehensive"}

state_dict: dict = {
    "species": species,
    "breed": breed or None,
    "age": age,
    "gender": gender_map[gender],
    "is_neutered": is_neutered,
    "weight": weight,
}

if coverage in coverage_map:
    state_dict["coverage_style"] = coverage_map[coverage]
if selected_insurers:
    state_dict["preferred_insurers"] = selected_insurers

health = {}
if illness_area.strip():
    health["frequent_illness_area"] = illness_area.strip()
if disease_history.strip():
    health["disease_surgery_history"] = disease_history.strip()
if health:
    state_dict["health_condition"] = health

# ── 메인: 입력 요약 + 결과 ────────────────────────────
col_info, col_result = st.columns([1, 2])

with col_info:
    st.subheader("입력 요약")
    st.markdown(f"**종**: {species}")
    st.markdown(f"**품종**: {breed or '-'}")
    st.markdown(f"**나이**: {age}세")
    st.markdown(f"**성별**: {gender}")
    st.markdown(f"**체중**: {weight}kg")
    st.markdown(f"**중성화**: {'예' if is_neutered else '아니오'}")
    if coverage != "선택 안함":
        st.markdown(f"**보장 스타일**: {coverage}")
    if selected_insurers:
        st.markdown(f"**선호 보험사**: {', '.join(selected_insurers)}")
    if illness_area.strip():
        st.markdown(f"**자주 아픈 부위**: {illness_area}")
    if disease_history.strip():
        st.markdown(f"**질병/수술 이력**: {disease_history}")

with col_result:
    st.subheader("추천 결과")

    if run_btn:
        if not breed:
            st.warning("품종을 입력해주세요.")
            st.stop()

        config = make_config(thread_id="streamlit_demo")
        with st.spinner("파이프라인 실행 중..."):
            result = graph.invoke(state_dict, config=config)

        # 질병 진단
        diseases = result.get("diseases", [])
        if diseases:
            st.markdown("#### 진단 질병")
            for d in diseases:
                st.markdown(
                    f"- **{d.name}** (발병률: {d.incidence_rate}, 시기: {d.onset_period})"
                )

        # 검증 결과
        validation = result.get("validation_result")
        if validation:
            st.markdown("#### 보험 상품 검증")
            policies = validation.get("selected_policies", [])
            for p in policies:
                score = p.get("suitability_score", p.get("score", "-"))
                st.markdown(
                    f"- **{p.get('product_name', '-')}** (적합도: {score}점) — {p.get('reason', '')}"
                )
            summary = validation.get("review_summary")
            if summary:
                st.info(summary)

        # 최종 메시지
        final = result.get("final_message", "")
        if final:
            st.markdown("#### 최종 추천 메시지")
            st.success(final)
    else:
        st.info("왼쪽에서 정보를 입력하고 '추천 실행' 버튼을 눌러주세요.")
