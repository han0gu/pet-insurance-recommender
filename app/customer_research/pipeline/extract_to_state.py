#!/usr/bin/env python3
"""
게시글에서 UserInputTemplateState 호환 시나리오를 추출하는 스크립트 (v2)

출력 디렉토리: data/interim/scenarios_v2_yaml/
출력 형식: YAML (meta + state)
여러 마리: 1 게시글 → N개 파일 ({article_id}_pet_{index}.yaml)

사용법:
    python pipeline/extract_to_state.py                  # 전체 실행
    python pipeline/extract_to_state.py --sample 5       # 샘플 5건만 테스트
    python pipeline/extract_to_state.py --ids 33968 37811  # 특정 게시글만 처리
"""

import argparse
import json
import os
import re
import signal
import sys
import time

import yaml
from pathlib import Path
from dotenv import load_dotenv
from langchain_upstage import ChatUpstage
from langchain_core.messages import HumanMessage, SystemMessage


class LLMTimeoutError(Exception):
    pass


def _timeout_handler(signum, frame):
    raise LLMTimeoutError("LLM 호출 타임아웃")


LLM_TIMEOUT_SEC = 60  # LLM 호출 최대 대기 시간

load_dotenv()

UPSTAGE_API_KEY = os.getenv("UPSTAGE_API_KEY")
if not UPSTAGE_API_KEY:
    raise ValueError("UPSTAGE_API_KEY가 .env 파일에 설정되어 있지 않습니다.")

llm = ChatUpstage(api_key=UPSTAGE_API_KEY, model="solar-pro")

# ============================================================
# 보험사 정식 명칭 매핑
# ============================================================
INSURER_ENUM = [
    "삼성화재해상보험",
    "KB손해보험",
    "현대해상화재보험",
    "DB손해보험",
    "메리츠화재해상보험",
    "한화손해보험",
    "롯데손해보험",
    "농협손해보험",
    "라이나손해보험",
    "캐롯손해보험",
    "마이브라운 반려동물전문보험",
]

# ============================================================
# LLM 시스템 프롬프트
# ============================================================
SYSTEM_PROMPT = """\
당신은 펫보험 네이버 카페 게시글을 분석하여 구조화된 데이터를 추출하는 전문가입니다.
주어진 게시글(본문 + 댓글)에서 아래 규칙에 따라 YAML 형식으로 정보를 추출하세요.

## 추출 규칙

### 메타 정보 (meta)
- is_pet_insurance_related: 펫보험과 직접 관련 없는 게시글(화재보험, 자동차보험, 일상 이야기, 뉴스 공유 등)은 false
- question_intent: 다음 중 하나 선택
  가입_적합성_확인 | 보험사_비교_추천 | 보장내용_확인 | 보험료_확인 | 보험금_청구_문의 | 일반_문의
- user_concerns: 해당하는 것 모두 선택 (배열)
  보험료 | 보장범위 | 슬개골 | 치주질환 | 피부질환 | 자기부담금 | 보상비율 | 보험금청구 | 갱신 | 보험사비교
- original_question_summary: 핵심 질문을 1-2문장으로 요약 (카페 규칙, 인사말 제외)

### 반려동물 정보 (pets 배열)
- 게시글에 여러 마리가 언급되면 각각 별도 항목으로 작성
- 한 마리만 언급되거나 구분이 불가능하면 pets 배열에 1개만 작성
- 각 항목의 필드:
  - species: "강아지" 또는 "고양이" (없으면 null)
  - breed: 품종명 (없으면 null)
  - age: 정수(세 단위). 개월 수는 반올림(6개월→1). 없으면 null
  - gender: "male" 또는 "female" (없으면 null)
  - is_neutered: true 또는 false (없으면 null)
  - weight: 정수(kg) (없으면 null)
  - health_condition:
      frequent_illness_area: 자주 아픈 부위 (없으면 null)
      disease_surgery_history: 질병/수술 이력 문자열 (없으면 null)
  - coverage_style: "minimal" 또는 "comprehensive" (추론 불가 시 null)
      comprehensive: "보장 넓게", "종합", "안심", "다 보장" 등의 표현이 있을 때
      minimal: "저렴하게", "기본만", "실속", "최소" 등의 표현이 있을 때
  - preferred_insurers: 보험사 정식 명칭 리스트 (없으면 null)
      메리츠, 펫퍼민트 → 메리츠화재해상보험
      삼성, 삼성화재 → 삼성화재해상보험
      현대, 현대해상 → 현대해상화재보험
      KB, KB손해, KB손보, 금쪽같은 → KB손해보험
      DB, DB손해, DB손보, 펫블리 → DB손해보험
      한화 → 한화손해보험
      롯데 → 롯데손해보험
      농협 → 농협손해보험
      라이나 → 라이나손해보험
      캐롯 → 캐롯손해보험
      마이브라운 → 마이브라운 반려동물전문보험

### 설계사 답변 (expert_advice)
- 댓글 중 "인스포유", "보험전문가", "공식전문가", "공식" 등이 포함된 답변 = 설계사 답변
- has_expert_reply: 설계사 답변 존재 여부
- expert_name: 설계사 닉네임 (없으면 null)
- reply_summary: 핵심 조언 2-3문장 요약. 광고, 서명, 카카오톡 링크, 금융소비자보호법 안내는 반드시 제외
- recommended_insurers: 설계사가 추천한 보험사 정식 명칭 배열. A/B/C 익명 처리된 경우 null
- coverage_advice: 보장 관련 조언 한 줄 요약 (없으면 null)
- price_info: 보험료 관련 정보 한 줄 요약 (없으면 null)

## 출력 형식 (반드시 아래 YAML 구조를 정확히 따르세요)

meta:
  article_id: "아이디"
  written_date: "날짜"
  is_pet_insurance_related: true
  question_intent: "의도"
  user_concerns:
    - 항목1
  original_question_summary: "요약"
  total_pets: 1
  expert_advice:
    has_expert_reply: false
    expert_name: null
    reply_summary: null
    recommended_insurers: null
    coverage_advice: null
    price_info: null
pets:
  - species: null
    breed: null
    age: null
    gender: null
    is_neutered: null
    weight: null
    health_condition:
      frequent_illness_area: null
      disease_surgery_history: null
    coverage_style: null
    preferred_insurers: null

## 주의사항
- 게시글에 명시되지 않은 정보는 반드시 null로 작성하세요. 절대 추측하지 마세요.
- YAML만 출력하세요. 설명, 주석, 코드 블록(```)은 포함하지 마세요.
- 들여쓰기는 2칸 스페이스를 사용하세요."""


# ============================================================
# 본문/댓글 전처리
# ============================================================
UI_NOISE = [
    "URL 복사",
    "URL이 복사되었습니다",
    "카페 캘린더",
    "제외하시겠습니까",
    "보내시겠습니까",
    "레이어팝업 닫기",
    "좋아요",
    "공유",
    "신고",
    "등록순",
    "최신순",
    "새로고침",
    "클린봇",
    "악성 댓글",
    "지금 가입하고",
    "멤버 리스트",
    "인기멤버",
    "일반멤버",
    "신입멤버",
    "1:1 채팅",
]

SKIP_EXACT = {"확인", "취소", ">", "\u200b", "0", "1", "2", "댓글"}


def clean_body(body: str) -> str:
    """본문에서 카페 UI 보일러플레이트 제거"""
    if "--------" in body:
        body = body.split("--------", 1)[-1]
    if "태그" in body:
        body = body.split("태그")[0]
    if "님의 게시글" in body:
        body = body.split("님의 게시글")[0]

    lines = []
    for line in body.split("\n"):
        stripped = line.strip()
        if not stripped:
            continue
        if stripped in SKIP_EXACT:
            continue
        if any(ui in stripped for ui in UI_NOISE):
            continue
        lines.append(stripped)

    return "\n".join(lines)


def format_comments(comments: list) -> str:
    """댓글 목록을 텍스트로 포매팅"""
    if not comments:
        return "댓글 없음"
    parts = []
    for c in comments:
        author = c.get("author", "")
        content = c.get("content", "").strip()
        if content:
            parts.append(f"[{author}]:\n{content}")
    return "\n\n".join(parts) if parts else "댓글 없음"


# ============================================================
# LLM 호출 및 YAML 파싱
# ============================================================
def extract_yaml_from_response(text: str) -> str:
    """LLM 응답에서 YAML 부분 추출"""
    m = re.search(r"```(?:yaml)?\s*\n(.*?)```", text, re.DOTALL)
    if m:
        return m.group(1).strip()
    return text.strip()


def call_llm(article: dict, max_retries: int = 2) -> dict | None:
    """LLM을 호출하여 구조화된 데이터 추출"""
    article_id = article.get("article_id", "")
    title = article.get("title", "")
    body = article.get("body", "")
    comments = article.get("comments", [])
    written_date = article.get("written_date", "")

    cleaned = clean_body(body)
    comments_text = format_comments(comments)

    user_message = (
        f"게시글 ID: {article_id}\n"
        f"작성일: {written_date}\n\n"
        f"제목: {title}\n\n"
        f"본문:\n{cleaned[:2000]}\n\n"
        f"댓글:\n{comments_text[:1500]}"
    )

    messages = [
        SystemMessage(content=SYSTEM_PROMPT),
        HumanMessage(content=user_message),
    ]

    for attempt in range(1, max_retries + 1):
        try:
            # 타임아웃 설정
            old_handler = signal.signal(signal.SIGALRM, _timeout_handler)
            signal.alarm(LLM_TIMEOUT_SEC)

            response = llm.invoke(messages)

            signal.alarm(0)  # 타임아웃 해제
            signal.signal(signal.SIGALRM, old_handler)

            yaml_text = extract_yaml_from_response(response.content)
            data = yaml.safe_load(yaml_text)

            if isinstance(data, dict) and "meta" in data:
                return data

            print(f"    [재시도 {attempt}] YAML 구조 불일치", flush=True)
        except LLMTimeoutError:
            signal.alarm(0)
            print(f"    [재시도 {attempt}] 타임아웃 ({LLM_TIMEOUT_SEC}초)", flush=True)
        except yaml.YAMLError as e:
            signal.alarm(0)
            print(f"    [재시도 {attempt}] YAML 파싱 오류: {e}", flush=True)
        except Exception as e:
            signal.alarm(0)
            print(f"    [재시도 {attempt}] LLM 호출 오류: {e}", flush=True)

        if attempt < max_retries:
            time.sleep(1)

    return None


# ============================================================
# 후처리 및 검증
# ============================================================
VALID_INTENTS = {
    "가입_적합성_확인",
    "보험사_비교_추천",
    "보장내용_확인",
    "보험료_확인",
    "보험금_청구_문의",
    "일반_문의",
}

VALID_CONCERNS = {
    "보험료",
    "보장범위",
    "슬개골",
    "치주질환",
    "피부질환",
    "자기부담금",
    "보상비율",
    "보험금청구",
    "갱신",
    "보험사비교",
}


def validate_and_fix(data: dict, article: dict) -> dict:
    """추출 결과를 검증하고 보정"""
    meta = data.get("meta", {})

    # article_id 보정
    meta["article_id"] = str(article.get("article_id", meta.get("article_id", "")))
    meta["written_date"] = article.get("written_date", meta.get("written_date", ""))

    # question_intent 검증
    if meta.get("question_intent") not in VALID_INTENTS:
        meta["question_intent"] = "일반_문의"

    # user_concerns 검증
    concerns = meta.get("user_concerns", [])
    if isinstance(concerns, list):
        meta["user_concerns"] = [c for c in concerns if c in VALID_CONCERNS]
    else:
        meta["user_concerns"] = []

    # expert_advice 기본값
    if "expert_advice" not in meta or meta["expert_advice"] is None:
        meta["expert_advice"] = {
            "has_expert_reply": False,
            "expert_name": None,
            "reply_summary": None,
            "recommended_insurers": None,
            "coverage_advice": None,
            "price_info": None,
        }

    # preferred_insurers 정규화
    pets = data.get("pets", [])
    if not isinstance(pets, list) or len(pets) == 0:
        pets = [{}]

    for pet in pets:
        insurers = pet.get("preferred_insurers")
        if isinstance(insurers, list):
            pet["preferred_insurers"] = [
                ins for ins in insurers if ins in INSURER_ENUM
            ] or None

        # health_condition 기본 구조 보장
        hc = pet.get("health_condition")
        if not isinstance(hc, dict):
            pet["health_condition"] = {
                "frequent_illness_area": None,
                "disease_surgery_history": None,
            }

        # age를 정수로 강제 변환
        age = pet.get("age")
        if age is not None:
            try:
                pet["age"] = int(age)
            except (ValueError, TypeError):
                pet["age"] = None

        # weight를 정수로 강제 변환
        weight = pet.get("weight")
        if weight is not None:
            try:
                pet["weight"] = int(weight)
            except (ValueError, TypeError):
                pet["weight"] = None

    meta["total_pets"] = len(pets)
    data["meta"] = meta
    data["pets"] = pets

    return data


# ============================================================
# 파일 저장
# ============================================================
def split_and_save(data: dict, output_dir: Path) -> int:
    """pets 배열을 개별 파일로 분리하여 저장. 저장된 파일 수 반환."""
    meta = data["meta"]
    pets = data["pets"]
    article_id = meta["article_id"]
    total_pets = meta["total_pets"]
    saved = 0

    for i, pet_state in enumerate(pets):
        pet_index = i + 1
        scenario = {
            "meta": {**meta, "pet_index": pet_index},
            "state": pet_state,
        }

        if total_pets > 1:
            filename = f"{article_id}_pet_{pet_index}.yaml"
        else:
            filename = f"{article_id}.yaml"

        filepath = output_dir / filename
        with open(filepath, "w", encoding="utf-8") as f:
            yaml.dump(
                scenario,
                f,
                allow_unicode=True,
                default_flow_style=False,
                sort_keys=False,
            )
        saved += 1

    return saved


# ============================================================
# 메인 실행
# ============================================================
def parse_args():
    parser = argparse.ArgumentParser(
        description="게시글에서 UserInputTemplateState 호환 시나리오 추출"
    )
    parser.add_argument(
        "--sample",
        type=int,
        default=0,
        help="테스트용 샘플 건수 (0이면 전체 실행)",
    )
    parser.add_argument(
        "--ids",
        nargs="+",
        default=[],
        help="특정 게시글 ID만 처리 (예: --ids 33968 37811)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="data/interim/scenarios_v2_yaml",
        help="출력 디렉토리 (기본: data/interim/scenarios_v2_yaml)",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="이미 처리된 게시글은 건너뛰고 이어서 실행",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    project_root = Path(__file__).resolve().parent.parent
    articles_dir = project_root / "data" / "raw" / "articles"
    output_dir = project_root / args.output_dir
    output_dir.mkdir(exist_ok=True)

    # 처리 대상 결정
    if args.ids:
        article_files = [articles_dir / f"{aid}.json" for aid in args.ids]
        article_files = [f for f in article_files if f.exists()]
    else:
        article_files = sorted(articles_dir.glob("*.json"))

    if args.sample > 0:
        # 다양한 케이스 포함을 위해 균등 간격 샘플링
        step = max(1, len(article_files) // args.sample)
        article_files = article_files[::step][: args.sample]

    # resume 모드: 이미 처리된 article_id 수집
    processed_ids = set()
    if args.resume:
        for existing in output_dir.glob("*.yaml"):
            # 파일명에서 article_id 추출 (예: 33968.yaml, 33968_pet_1.yaml)
            stem = existing.stem
            aid = stem.split("_pet_")[0]
            processed_ids.add(aid)
        print(f"이미 처리된 게시글: {len(processed_ids)}개 (건너뜀)\n", flush=True)

    total = len(article_files)
    print(f"총 {total}개 게시글 처리 시작...", flush=True)
    print(f"출력 디렉토리: {output_dir}/\n", flush=True)

    stats = {"success": 0, "not_related": 0, "failed": 0, "skipped": 0, "total_files": 0}

    for i, article_file in enumerate(article_files, 1):
        with open(article_file, "r", encoding="utf-8") as f:
            article = json.load(f)

        article_id = article.get("article_id", "")

        # resume 모드: 이미 처리된 건 건너뜀
        if args.resume and str(article_id) in processed_ids:
            stats["skipped"] += 1
            continue

        print(f"[{i}/{total}] {article_id} 추출 중...", flush=True)

        # LLM 호출
        data = call_llm(article)
        if data is None:
            print(f"  → 추출 실패", flush=True)
            stats["failed"] += 1
            continue

        # 후처리 및 검증
        data = validate_and_fix(data, article)

        # 펫보험 비관련 게시글 처리
        is_related = data["meta"].get("is_pet_insurance_related", False)
        if not is_related:
            print(f"  → 펫보험 비관련 (건너뜀)", flush=True)
            stats["not_related"] += 1
            # 비관련도 기록 (추후 분석용)
            saved = split_and_save(data, output_dir)
            stats["total_files"] += saved
            continue

        # 저장
        saved = split_and_save(data, output_dir)
        stats["total_files"] += saved
        pet_count = data["meta"]["total_pets"]
        summary = data["meta"].get("original_question_summary", "")[:60]
        print(f"  → 완료 (pet: {pet_count}개, file: {saved}개) {summary}", flush=True)
        stats["success"] += 1

        # Rate limit 방지
        time.sleep(0.5)

    print(f"\n{'='*60}", flush=True)
    print(f"처리 완료!", flush=True)
    print(f"  성공 (관련): {stats['success']}개", flush=True)
    print(f"  비관련:      {stats['not_related']}개", flush=True)
    print(f"  실패:        {stats['failed']}개", flush=True)
    if args.resume:
        print(f"  건너뜀:      {stats['skipped']}개 (이미 처리)", flush=True)
    print(f"  생성 파일:   {stats['total_files']}개", flush=True)
    print(f"  출력 위치:   {output_dir}/", flush=True)


if __name__ == "__main__":
    main()
