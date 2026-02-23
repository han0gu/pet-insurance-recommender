"""
데이터 정제 파이프라인 패키지

Stage 1: extract_scenarios.py    - 게시글에서 rule-based 시나리오 추출
         summarize_with_llm.py   - LLM 기반 질문 요약
Stage 3: extract_to_state.py     - LLM 기반 UserInputTemplateState 구조화 추출
Stage 4: fill_required_fields.py - 필수 필드 null 값 통계 기반 보정
"""
