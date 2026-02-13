"""반려동물 보험 추천 시스템 - Streamlit 데모 UI"""

from pathlib import Path

import yaml
import streamlit as st

from app.agents.orchestrator.orchestrator_graph import run_orchestration
from app.agents.user_input_template_agent.utils.cli import make_config

# ── 설정 ──────────────────────────────────────────────
SAMPLES_DIR = Path("app/agents/user_input_template_agent/samples/naver")
GENDER_KR = {"male": "수컷", "female": "암컷"}

st.set_page_config(page_title="반려동물 보험 추천", layout="wide")
st.title("반려동물 보험 추천 시스템")


# ── YAML 파일 목록 ────────────────────────────────────
yaml_files = sorted(SAMPLES_DIR.glob("*.yaml"))
if not yaml_files:
    st.error("샘플 YAML 파일을 찾을 수 없습니다.")
    st.stop()

# ── 사이드바: 샘플 선택 ──────────────────────────────
with st.sidebar:
    st.header("샘플 데이터 선택")
    selected = st.selectbox(
        "YAML 파일",
        yaml_files,
        format_func=lambda p: p.stem,
    )
    run_btn = st.button("추천 실행", type="primary", use_container_width=True)

# ── YAML 파싱 ─────────────────────────────────────────
raw = yaml.safe_load(selected.read_text())
state_data = raw.get("state", raw)
meta = raw.get("meta", {})

# ── 메인: 입력 정보 표시 ──────────────────────────────
col_info, col_result = st.columns([1, 2])

with col_info:
    st.subheader("입력 정보")
    st.markdown(f"**종**: {state_data.get('species', '-')}")
    st.markdown(f"**품종**: {state_data.get('breed', '-')}")
    st.markdown(f"**나이**: {state_data.get('age', '-')}세")
    st.markdown(f"**성별**: {GENDER_KR.get(state_data.get('gender', ''), '-')}")
    st.markdown(f"**체중**: {state_data.get('weight', '-')}kg")

    neutered = state_data.get("is_neutered")
    if neutered is not None:
        st.markdown(f"**중성화**: {'예' if neutered else '아니오'}")

    insurers = state_data.get("preferred_insurers")
    if insurers:
        st.markdown(f"**선호 보험사**: {', '.join(insurers)}")

    if meta.get("original_question_summary"):
        st.markdown("---")
        st.caption("원본 질문 요약")
        st.write(meta["original_question_summary"])

# ── 메인: 실행 결과 ───────────────────────────────────
with col_result:
    st.subheader("추천 결과")

    if run_btn:
        config = make_config(thread_id="streamlit_demo")
        with st.spinner("파이프라인 실행 중..."):
            result = run_orchestration(str(selected), config=config)

        # 질병 진단
        diseases = result.get("diseases", [])
        if diseases:
            st.markdown("#### 진단 질병")
            for d in diseases:
                st.markdown(f"- **{d.name}** (발병률: {d.incidence_rate}, 시기: {d.onset_period})")

        # 검증 결과
        validation = result.get("validation_result")
        if validation:
            st.markdown("#### 보험 상품 검증")
            policies = validation.get("selected_policies", [])
            for p in policies:
                score = p.get("suitability_score", p.get("score", "-"))
                st.markdown(f"- **{p.get('product_name', '-')}** (적합도: {score}점) — {p.get('reason', '')}")
            summary = validation.get("review_summary")
            if summary:
                st.info(summary)

        # 최종 메시지
        final = result.get("final_message", "")
        if final:
            st.markdown("#### 최종 추천 메시지")
            st.success(final)
    else:
        st.info("왼쪽에서 샘플을 선택하고 '추천 실행' 버튼을 눌러주세요.")
