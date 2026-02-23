#!/usr/bin/env python3
"""
Upstage LLM을 사용하여 펫보험 질문 시나리오를 요약하는 스크립트
"""

import json
import os
import time
from pathlib import Path
from dotenv import load_dotenv
from langchain_upstage import ChatUpstage
from langchain_core.messages import HumanMessage, SystemMessage

# .env 파일 로드
load_dotenv()

# Upstage API 설정
UPSTAGE_API_KEY = os.getenv("UPSTAGE_API_KEY")

if not UPSTAGE_API_KEY:
    raise ValueError("UPSTAGE_API_KEY가 .env 파일에 설정되어 있지 않습니다.")

# LLM 초기화
llm = ChatUpstage(
    api_key=UPSTAGE_API_KEY,
    model="solar-pro",  # 또는 solar-mini
)

SYSTEM_PROMPT = """당신은 펫보험 관련 사용자 질문을 분석하고 요약하는 전문가입니다.

주어진 게시글 내용을 분석하여 다음 정보를 추출하고 간결하게 요약해주세요:

1. 사용자의 핵심 질문/요청 (1-2문장)
2. 반려동물 정보 (종류, 품종, 나이 등)
3. 관심 보험사/상품
4. 주요 관심사 또는 우려사항

응답 형식:
- 핵심 질문을 먼저 작성
- 부가 정보는 괄호 안에 간략히 표기
- 전체 150자 이내로 작성
- 불필요한 인사말, 카페 규칙 등은 제외"""


def get_original_body(article_id: str) -> str:
    """원본 게시글 본문 가져오기"""
    project_root = Path(__file__).resolve().parent.parent
    articles_dir = project_root / "data" / "raw" / "articles"
    article_file = articles_dir / f"{article_id}.json"
    
    if article_file.exists():
        with open(article_file, "r", encoding="utf-8") as f:
            article = json.load(f)
        return article.get("body", "")
    return ""


def summarize_with_llm(title: str, body: str) -> str:
    """LLM을 사용하여 질문 요약 생성"""
    # 본문에서 템플릿 제거
    if "--------" in body:
        body = body.split("--------", 1)[-1]
    
    # 서명/광고 제거
    body = body.split("태그")[0] if "태그" in body else body
    body = body.split("님의 게시글")[0] if "님의 게시글" in body else body
    
    # 줄바꿈 정리
    body = " ".join([line.strip() for line in body.split("\n") if line.strip()])
    
    # 입력 텍스트 준비
    input_text = f"제목: {title}\n\n본문:\n{body[:1500]}"  # 토큰 제한 고려
    
    messages = [
        SystemMessage(content=SYSTEM_PROMPT),
        HumanMessage(content=input_text),
    ]
    
    try:
        response = llm.invoke(messages)
        return response.content.strip()
    except Exception as e:
        print(f"LLM 호출 오류: {e}")
        return None


def process_scenarios():
    """모든 시나리오 파일 처리"""
    project_root = Path(__file__).resolve().parent.parent
    scenarios_dir = project_root / "data" / "interim" / "scenarios"
    articles_dir = project_root / "data" / "raw" / "articles"
    
    # 처리할 시나리오 파일 목록
    scenario_files = sorted(scenarios_dir.glob("SC_*.json"))
    total = len(scenario_files)
    
    print(f"총 {total}개 시나리오 처리 시작...", flush=True)
    
    for i, scenario_file in enumerate(scenario_files, 1):
        with open(scenario_file, "r", encoding="utf-8") as f:
            scenario = json.load(f)
        
        article_id = scenario.get("source_article_id")
        
        # 원본 게시글 가져오기
        article_file = articles_dir / f"{article_id}.json"
        if not article_file.exists():
            print(f"[{i}/{total}] {article_id}: 원본 게시글 없음, 건너뜀", flush=True)
            continue
        
        with open(article_file, "r", encoding="utf-8") as f:
            article = json.load(f)
        
        title = article.get("title", "")
        body = article.get("body", "")
        
        print(f"[{i}/{total}] {article_id} 요약 중...", flush=True)
        
        # LLM 요약 생성
        llm_summary = summarize_with_llm(title, body)
        
        if llm_summary:
            # 시나리오 업데이트
            scenario["original_question_summary"] = llm_summary
            
            with open(scenario_file, "w", encoding="utf-8") as f:
                json.dump(scenario, f, ensure_ascii=False, indent=2)
            
            print(f"    -> 완료: {llm_summary[:50]}...", flush=True)
        else:
            print(f"    -> 요약 실패, 기존 유지", flush=True)
        
        # Rate limit 방지
        time.sleep(0.5)
    
    print("\n모든 시나리오 처리 완료!", flush=True)
    
    # scenarios_summary.json도 업데이트
    update_summary_file(scenarios_dir)


def update_summary_file(scenarios_dir: Path):
    """scenarios_summary.json 업데이트"""
    summary_file = scenarios_dir / "scenarios_summary.json"
    
    if not summary_file.exists():
        return
    
    with open(summary_file, "r", encoding="utf-8") as f:
        summary = json.load(f)
    
    # 개별 시나리오에서 다시 로드
    updated_scenarios = []
    for scenario_file in sorted(scenarios_dir.glob("SC_*.json")):
        with open(scenario_file, "r", encoding="utf-8") as f:
            updated_scenarios.append(json.load(f))
    
    summary["scenarios"] = updated_scenarios
    
    with open(summary_file, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    
    print("scenarios_summary.json 업데이트 완료!")


if __name__ == "__main__":
    process_scenarios()
