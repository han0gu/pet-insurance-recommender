#!/usr/bin/env python3
"""
펫보험 관련 질문 글에서 사용자 시나리오를 추출하는 스크립트
"""

import json
import os
import re
from pathlib import Path
from typing import Optional


def is_pet_insurance_question(article: dict) -> bool:
    """펫보험 관련 질문 글인지 판별"""
    title = article.get("title", "")
    body = article.get("body", "")

    # 펫보험과 무관한 키워드 (제외 대상)
    exclude_keywords = [
        "장훈",
        "국적",
        "야구",
        "정치",
        "선거",
        "대통령",
        "장례",
        "전남친",
        "이혼",
        "결혼",
        "연예",
        "드라마",
        "영화",
    ]

    for keyword in exclude_keywords:
        if keyword in title or keyword in body[:500]:
            return False

    # 펫보험 질문 패턴
    question_patterns = [
        r"어떤가요\??",
        r"괜찮[은나을]가요\??",
        r"괜찮을까요\??",
        r"어때요\??",
        r"알려주세요",
        r"도와주세요",
        r"확인해주세요",
        r"문의드립니다",
        r"궁금합니다",
        r"궁금해서",
        r"추천해\s*주세요",
        r"추천\s*부탁",
        r"가입해도\s*될까요",
        r"가입\s*고민",
        r"비교\s*해주세요",
        r"\[알려주세요\]",
        r"\[도와주세요\]",
        r"\[확인해주세요\]",
    ]

    # 펫보험 관련 키워드
    pet_keywords = [
        "펫보험",
        "반려동물보험",
        "강아지보험",
        "고양이보험",
        "메리츠",
        "삼성화재",
        "현대해상",
        "KB손해",
        "DB손해",
        "KB손보",
        "펫퍼민트",
        "금쪽같은",
    ]

    # 반려동물 관련 키워드
    pet_animal_keywords = [
        "강아지",
        "댕댕이",
        "반려견",
        "고양이",
        "냥이",
        "반려묘",
        "말티즈",
        "푸들",
        "토이푸들",
        "말티푸",
        "비숑",
        "포메라니안",
        "시츄",
        "치와와",
        "믹스견",
        "대형견",
        "소형견",
        "중형견",
    ]

    text = title + " " + body

    # 펫보험 관련 키워드 확인
    has_pet_keyword = any(kw in text for kw in pet_keywords)
    has_pet_animal = any(kw in text for kw in pet_animal_keywords)

    # 질문 패턴 확인
    has_question = any(re.search(pattern, text) for pattern in question_patterns)

    # 뉴스/정보 공유 글 제외 (링크만 있고 질문이 없는 글)
    if "heraldcorp.com" in body or "cstimes.com" in body or "dt.co.kr" in body:
        if not has_question:
            return False

    return has_pet_keyword and (has_pet_animal or has_question) and has_question


def extract_pet_info(body: str) -> dict:
    """본문에서 반려동물 정보 추출"""
    pet_info = {
        "animal_type": None,
        "breed": None,
        "age": None,
        "weight": None,
        "special_condition": None,
    }

    # 동물 종류
    if any(kw in body for kw in ["고양이", "냥이", "반려묘", "묘"]):
        pet_info["animal_type"] = "고양이"
    elif any(kw in body for kw in ["강아지", "댕댕이", "반려견", "개"]):
        pet_info["animal_type"] = "강아지"

    # 품종 추출
    breeds = {
        "말티즈": ["말티즈"],
        "토이푸들": ["토이푸들", "토이 푸들"],
        "말티푸": ["말티푸"],
        "비숑": ["비숑", "비숑프리제"],
        "포메라니안": ["포메라니안", "포메"],
        "시츄": ["시츄"],
        "치와와": ["치와와"],
        "골든리트리버": ["골든리트리버", "골든 리트리버"],
        "믹스견": ["믹스견", "믹스"],
        "코리안숏헤어": ["코숏", "코리안숏헤어"],
    }

    for breed_name, keywords in breeds.items():
        if any(kw in body for kw in keywords):
            pet_info["breed"] = breed_name
            break

    # 나이 추출
    age_patterns = [
        r"(\d+)\s*세",
        r"(\d+)\s*살",
        r"(\d+)\s*개월",
    ]

    for pattern in age_patterns:
        match = re.search(pattern, body)
        if match:
            if "개월" in pattern:
                pet_info["age"] = f"{match.group(1)}개월"
            else:
                pet_info["age"] = f"{match.group(1)}세"
            break

    # 체중 추출
    weight_match = re.search(r"(\d+)\s*키로", body)
    if weight_match:
        pet_info["weight"] = f"{weight_match.group(1)}kg"

    # 특수 상황
    if "유기견" in body or "구조" in body:
        pet_info["special_condition"] = "유기견_입양"
    elif "중성화" in body:
        pet_info["special_condition"] = "중성화_완료"

    return pet_info


def extract_user_concerns(body: str) -> list:
    """사용자의 주요 관심사/우려사항 추출"""
    concerns = []

    concern_keywords = {
        "보험료": ["보험료", "가격", "비용", "금액"],
        "보장범위": ["보장", "보장범위", "보장내용", "담보"],
        "슬개골": ["슬개골", "슬관절"],
        "치주질환": ["치주", "스케일링", "발치", "치과"],
        "피부질환": ["피부", "아토피", "아포퀠"],
        "자기부담금": ["자기부담금", "자부담"],
        "보상비율": ["보상비율", "보상율", "90%", "80%"],
        "보험금청구": ["청구", "자동청구", "보험금"],
        "갱신": ["갱신", "갱신형"],
        "보험사비교": ["비교", "어디가", "추천"],
    }

    for concern, keywords in concern_keywords.items():
        if any(kw in body for kw in keywords):
            concerns.append(concern)

    return concerns


def extract_target_insurers(body: str) -> list:
    """관심 보험사 추출"""
    insurers = []

    insurer_keywords = {
        "메리츠화재": ["메리츠", "펫퍼민트"],
        "삼성화재": ["삼성화재", "삼성"],
        "현대해상": ["현대해상", "현대"],
        "KB손해보험": ["KB손해", "KB손보", "KB금쪽"],
        "DB손해보험": ["DB손해", "DB손보"],
    }

    for insurer, keywords in insurer_keywords.items():
        if any(kw in body for kw in keywords):
            insurers.append(insurer)

    return insurers


def extract_question_intent(title: str, body: str) -> str:
    """질문 의도 분류"""
    text = title + " " + body

    if any(kw in text for kw in ["가입해도", "가입 고민", "가입할까", "들어도"]):
        return "가입_적합성_확인"
    elif any(kw in text for kw in ["비교", "추천", "어디가", "뭐가 좋"]):
        return "보험사_비교_추천"
    elif any(kw in text for kw in ["보장", "담보", "보장내용"]):
        return "보장내용_확인"
    elif any(kw in text for kw in ["보험료", "가격", "비용"]):
        return "보험료_확인"
    elif any(kw in text for kw in ["청구", "보험금"]):
        return "보험금_청구_문의"
    else:
        return "일반_문의"


def create_scenario_from_article(article: dict) -> Optional[dict]:
    """게시글에서 사용자 시나리오 생성"""
    if not is_pet_insurance_question(article):
        return None

    title = article.get("title", "")
    body = article.get("body", "")

    # 본문에서 실제 질문 부분 추출 (템플릿 제외)
    # "--------" 이후가 실제 질문 내용
    if "--------" in body:
        body = body.split("--------", 1)[-1]

    pet_info = extract_pet_info(body)
    concerns = extract_user_concerns(body)
    target_insurers = extract_target_insurers(body)
    question_intent = extract_question_intent(title, body)

    # 시나리오 생성
    scenario = {
        "scenario_id": f"SC_{article['article_id']}",
        "source_article_id": article["article_id"],
        "question_intent": question_intent,
        "pet_info": pet_info,
        "target_insurers": target_insurers,
        "user_concerns": concerns,
        "original_question_summary": extract_question_summary(title, body),
        "written_date": article.get("written_date", ""),
    }

    return scenario


def extract_question_summary(title: str, body: str) -> str:
    """질문 요약 추출"""
    # 제목에서 태그 제거
    title = re.sub(
        r"◈펫보험|말머리|\[알려주세요\]|\[도와주세요\]|\[확인해주세요\]", "", title
    ).strip()

    # 본문에서 핵심 질문 추출
    if "--------" in body:
        body = body.split("--------", 1)[-1]

    # 서명/광고 제거
    body = body.split("태그")[0] if "태그" in body else body
    body = body.split("님의 게시글")[0] if "님의 게시글" in body else body

    # 줄바꿈 정리
    lines = [line.strip() for line in body.split("\n") if line.strip()]
    summary_lines = []
    for line in lines[:10]:  # 처음 10줄만
        if len(line) > 5 and not line.startswith("※"):
            summary_lines.append(line)

    summary = " ".join(summary_lines)[:300]  # 300자로 제한

    return f"{title} - {summary}" if summary else title


def main():
    project_root = Path(__file__).resolve().parent.parent
    articles_dir = project_root / "data" / "raw" / "articles"
    scenarios_dir = project_root / "data" / "interim" / "scenarios"

    # 시나리오 폴더 생성
    scenarios_dir.mkdir(exist_ok=True)

    scenarios = []
    skipped = []

    # 모든 게시글 처리
    for article_file in sorted(articles_dir.glob("*.json")):
        with open(article_file, "r", encoding="utf-8") as f:
            article = json.load(f)

        scenario = create_scenario_from_article(article)

        if scenario:
            scenarios.append(scenario)
            # 개별 시나리오 파일 저장
            scenario_file = scenarios_dir / f"{scenario['scenario_id']}.json"
            with open(scenario_file, "w", encoding="utf-8") as f:
                json.dump(scenario, f, ensure_ascii=False, indent=2)
        else:
            skipped.append(
                {
                    "article_id": article["article_id"],
                    "title": article.get("title", ""),
                }
            )

    # 전체 시나리오 요약 저장
    summary = {
        "total_articles": len(list(articles_dir.glob("*.json"))),
        "total_scenarios": len(scenarios),
        "skipped_count": len(skipped),
        "scenarios": scenarios,
    }

    with open(scenarios_dir / "scenarios_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    # 제외된 글 목록 저장 (디버깅용)
    with open(scenarios_dir / "skipped_articles.json", "w", encoding="utf-8") as f:
        json.dump(skipped, f, ensure_ascii=False, indent=2)

    print(f"총 게시글: {summary['total_articles']}개")
    print(f"추출된 시나리오: {summary['total_scenarios']}개")
    print(f"제외된 게시글: {summary['skipped_count']}개")
    print(f"시나리오 저장 위치: {scenarios_dir}")


if __name__ == "__main__":
    main()
