#!/usr/bin/env python3
"""customer_research 모듈 통합 테스트"""

import sys
import random
from pathlib import Path

PASS = 0
FAIL = 0
ERRORS = []


def report(name: str, ok: bool, detail: str = ""):
    global PASS, FAIL
    if ok:
        PASS += 1
        print(f"  [PASS] {name}")
    else:
        FAIL += 1
        msg = f"  [FAIL] {name}" + (f" — {detail}" if detail else "")
        print(msg)
        ERRORS.append(msg)


# ================================================================
# 1. feature 모듈 import 테스트
# ================================================================
print("\n=== 1. feature 모듈 import ===")

try:
    from app.customer_research.feature.http_client import fetch_html, fetch_html_safe, _default_headers
    report("http_client import", True)
except Exception as e:
    report("http_client import", False, str(e))

try:
    from app.customer_research.feature.naver_cafe_parser import (
        get_article_titles_from_html,
        get_next_page_url,
        _article_id_from_href,
        _is_notice_title,
        _strip_comment_suffix,
    )
    report("naver_cafe_parser import", True)
except Exception as e:
    report("naver_cafe_parser import", False, str(e))

try:
    from app.customer_research.feature.naver_article_parser import (
        parse_article_html,
        save_article_json,
        _extract_body,
        _extract_title,
        _extract_comments,
        _extract_written_date,
    )
    report("naver_article_parser import", True)
except Exception as e:
    report("naver_article_parser import", False, str(e))


# ================================================================
# 2. pipeline 모듈 import 테스트
# ================================================================
print("\n=== 2. pipeline 모듈 import ===")

try:
    from app.customer_research.pipeline.extract_scenarios import (
        is_pet_insurance_question,
        extract_pet_info,
        extract_user_concerns,
        extract_target_insurers,
        extract_question_intent,
        create_scenario_from_article,
        extract_question_summary,
    )
    report("extract_scenarios import", True)
except Exception as e:
    report("extract_scenarios import", False, str(e))

try:
    from app.customer_research.pipeline.fill_required_fields import (
        weighted_choice,
        fill_breed,
        fill_age,
        fill_gender,
        fill_is_neutered,
        fill_weight,
        process_scenario,
        DOG_BREEDS,
        CAT_BREEDS,
        DOG_AGE_DIST,
        CAT_AGE_DIST,
    )
    report("fill_required_fields import", True)
except Exception as e:
    report("fill_required_fields import", False, str(e))

# LLM 기반 모듈은 UPSTAGE_API_KEY 필요
print("\n=== 2-1. LLM 모듈 import (API KEY 필요) ===")

try:
    from app.customer_research.pipeline.extract_to_state import (
        validate_and_fix,
        extract_yaml_from_response,
        clean_body,
        format_comments,
        VALID_INTENTS,
        VALID_CONCERNS,
    )
    report("extract_to_state import", True)
    extract_to_state_available = True
except Exception as e:
    report("extract_to_state import", False, f"(예상된 실패) {e}")
    extract_to_state_available = False

try:
    from app.customer_research.pipeline.summarize_with_llm import summarize_with_llm
    report("summarize_with_llm import", True)
except Exception as e:
    report("summarize_with_llm import", False, f"(예상된 실패) {e}")


# ================================================================
# 3. http_client 기능 테스트
# ================================================================
print("\n=== 3. http_client 기능 테스트 ===")

headers = _default_headers()
report("_default_headers 반환", isinstance(headers, dict) and "User-Agent" in headers)

html, err = fetch_html_safe("https://invalid-url-for-test-12345.com", timeout=3)
report("fetch_html_safe 잘못된 URL", html is None and err is not None,
       f"html={html}, err={err}")


# ================================================================
# 4. naver_cafe_parser 기능 테스트
# ================================================================
print("\n=== 4. naver_cafe_parser 기능 테스트 ===")

# _article_id_from_href
report("article_id: /articles/64944",
       _article_id_from_href("/articles/64944") == "64944")
report("article_id: /article/12345",
       _article_id_from_href("/article/12345") == "12345")
report("article_id: 없음",
       _article_id_from_href("/menus/41") is None)

# _is_notice_title
report("공지 판별: [공지] 제목", _is_notice_title("[공지] 중요 공지"))
report("공지 판별: 일반 제목", not _is_notice_title("펫보험 추천해주세요"))

# _strip_comment_suffix
report("댓글 수 제거: 제목 댓글 3",
       _strip_comment_suffix("제목 댓글 3") == "제목")
report("댓글 수 제거: 제목(5)",
       _strip_comment_suffix("제목(5)") == "제목")
report("댓글 수 제거: 일반 제목",
       _strip_comment_suffix("일반 제목") == "일반 제목")

# get_article_titles_from_html
sample_html = """
<html><body>
<div class="notice" data-article-type="notice">
  <a href="/articles/100">공지사항</a>
</div>
<a href="/articles/200">펫보험 가입 질문이요</a>
<a href="/articles/300">강아지 보험 추천 부탁드립니다</a>
<a href="/menus/41">게시판</a>
</body></html>
"""
titles = get_article_titles_from_html(sample_html, base_url="https://cafe.naver.com")
report("게시글 목록 추출 (공지 제외)",
       len(titles) >= 2,
       f"추출 수: {len(titles)}, 내용: {titles}")

# 추출된 항목에 글번호가 있는지 확인
if titles:
    first = titles[0]
    report("글번호 추출 확인", first[0] is not None, f"id={first[0]}")

# get_next_page_url
next_html = '<html><body><a href="/page2" rel="next">다음</a></body></html>'
next_url = get_next_page_url(next_html, base_url="https://cafe.naver.com")
report("다음 페이지 URL 추출",
       next_url is not None and "page2" in next_url,
       f"url={next_url}")

no_next_html = '<html><body><a href="/articles/123">게시글</a></body></html>'
report("다음 페이지 없음",
       get_next_page_url(no_next_html) is None)


# ================================================================
# 5. naver_article_parser 기능 테스트
# ================================================================
print("\n=== 5. naver_article_parser 기능 테스트 ===")

article_html = """
<html><body>
<h1 class="tit_article">말티즈 펫보험 추천 부탁드립니다</h1>
<div class="app_content">
질문 글 작성 전 필독 사항
안녕하세요, 3살 말티즈 키우고 있는데 펫보험 가입하려고 합니다.
메리츠랑 삼성화재 중에 고민 중인데 추천해주세요.
조언도 함께 부탁드릴게요
</div>
<span>2024.03.15. 14:30</span><span>조회 500</span>
<div class="comment_nickname">전문가A</div>
<div class="text_comment">메리츠 추천합니다.</div>
<div class="comment_nickname">일반인B</div>
<div class="text_comment">삼성화재도 좋아요.</div>
<a href="/articles/44443">link</a>
</body></html>
"""

parsed = parse_article_html(article_html, article_url="https://cafe.naver.com/articles/44443")

report("article_id 추출", parsed["article_id"] == "44443")
report("제목 추출", "말티즈" in parsed["title"] and "펫보험" in parsed["title"],
       f"title={parsed['title']}")
report("본문 추출 (마커 기반)",
       "필독 사항" in parsed["body"] and "조언도" in parsed["body"],
       f"body_len={len(parsed['body'])}")
report("댓글 추출",
       len(parsed["comments"]) >= 2,
       f"comments={len(parsed['comments'])}")
report("작성일 추출",
       "2024.03.15" in parsed["written_date"],
       f"date={parsed['written_date']}")


# ================================================================
# 6. extract_scenarios (Stage 1) 기능 테스트
# ================================================================
print("\n=== 6. extract_scenarios (Stage 1) 기능 테스트 ===")

# is_pet_insurance_question
article_yes = {
    "title": "강아지 펫보험 추천해주세요",
    "body": "말티즈 3살인데 펫보험 가입하려고 합니다. 어떤 보험이 좋을까요? 추천해주세요",
}
report("펫보험 질문 판별: 관련 글", is_pet_insurance_question(article_yes))

article_no_politics = {
    "title": "대통령 관련 뉴스",
    "body": "정치 관련 내용입니다",
}
report("펫보험 질문 판별: 정치 글 제외", not is_pet_insurance_question(article_no_politics))

article_no_question = {
    "title": "펫보험 뉴스",
    "body": "펫보험 관련 기사가 나왔습니다. 강아지 보험 시장이 커지고 있습니다. heraldcorp.com/link",
}
report("펫보험 질문 판별: 뉴스 링크 제외", not is_pet_insurance_question(article_no_question))

# extract_pet_info
pet_info = extract_pet_info("3살 말티즈 강아지를 키우고 있어요. 5키로입니다. 중성화 완료했어요.")
report("동물 종류 추출", pet_info["animal_type"] == "강아지")
report("품종 추출", pet_info["breed"] == "말티즈")
report("나이 추출", pet_info["age"] == "3세",
       f"코드는 '살'→'세'로 정규화, 실제={pet_info['age']}")
report("체중 추출", pet_info["weight"] == "5kg")
report("특수 상황 추출", pet_info["special_condition"] == "중성화_완료")

pet_info_cat = extract_pet_info("2세 코숏 고양이입니다")
report("고양이 종류 추출", pet_info_cat["animal_type"] == "고양이")
report("코숏 품종 추출", pet_info_cat["breed"] == "코리안숏헤어")

pet_info_months = extract_pet_info("6개월 된 강아지입니다")
report("개월 단위 나이 추출", pet_info_months["age"] == "6개월")

pet_info_rescue = extract_pet_info("유기견 입양한 강아지")
report("유기견 특수 상황", pet_info_rescue["special_condition"] == "유기견_입양")

# extract_user_concerns
concerns = extract_user_concerns("보험료가 궁금하고 슬개골 보장도 확인하고 싶어요")
report("관심사 추출: 보험료", "보험료" in concerns)
report("관심사 추출: 슬개골", "슬개골" in concerns)

concerns2 = extract_user_concerns("자기부담금이 없는 보험 추천해주세요. 비교해서 알려주세요.")
report("관심사 추출: 자기부담금", "자기부담금" in concerns2)
report("관심사 추출: 보험사비교", "보험사비교" in concerns2)

# extract_target_insurers
insurers = extract_target_insurers("메리츠랑 삼성화재 중에 고민 중입니다")
report("보험사 추출: 메리츠화재", "메리츠화재" in insurers)
report("보험사 추출: 삼성화재", "삼성화재" in insurers)

insurers2 = extract_target_insurers("KB손해보험이랑 현대해상 비교해주세요")
report("보험사 추출: KB손해보험", "KB손해보험" in insurers2)
report("보험사 추출: 현대해상", "현대해상" in insurers2)

# extract_question_intent
report("의도: 가입 적합성",
       extract_question_intent("가입해도 될까요", "") == "가입_적합성_확인")
report("의도: 보험사 비교",
       extract_question_intent("추천해주세요", "비교해서 알려주세요") == "보험사_비교_추천")
report("의도: 보장내용",
       extract_question_intent("", "보장내용 확인 부탁드립니다") == "보장내용_확인")
report("의도: 보험료",
       extract_question_intent("보험료 문의", "") == "보험료_확인")
report("의도: 보험금 청구",
       extract_question_intent("보험금 청구 방법", "") == "보험금_청구_문의")
report("의도: 일반 문의",
       extract_question_intent("안녕하세요", "질문 있습니다") == "일반_문의")

# create_scenario_from_article
full_article = {
    "article_id": "12345",
    "title": "강아지 펫보험 추천해주세요",
    "body": "3살 말티즈 키우고 있습니다. 보험료가 궁금해요. 메리츠 보험 추천해주세요.",
    "written_date": "2024.01.15. 14:30",
}
scenario = create_scenario_from_article(full_article)
report("시나리오 생성", scenario is not None)
if scenario:
    report("시나리오 ID", scenario["scenario_id"] == "SC_12345")
    report("시나리오 의도", scenario["question_intent"] in [
        "보험사_비교_추천", "보험료_확인", "가입_적합성_확인", "보장내용_확인",
        "보험금_청구_문의", "일반_문의"])
    report("시나리오 반려동물 정보", scenario["pet_info"]["breed"] == "말티즈")
    report("시나리오 보험사", "메리츠화재" in scenario["target_insurers"])

# 제외 케이스: 비관련 글
non_article = {
    "article_id": "99999",
    "title": "오늘 날씨 좋네요",
    "body": "산책하기 좋은 날입니다",
}
report("비관련 글 제외", create_scenario_from_article(non_article) is None)


# ================================================================
# 7. fill_required_fields (Stage 4) 기능 테스트
# ================================================================
print("\n=== 7. fill_required_fields (Stage 4) 기능 테스트 ===")

random.seed(42)

# weighted_choice
choices = {"A": 80, "B": 20}
results = [weighted_choice(choices) for _ in range(100)]
report("weighted_choice 분포 합리성",
       results.count("A") > results.count("B"),
       f"A={results.count('A')}, B={results.count('B')}")

# fill_breed
report("fill_breed: 기존 값 유지",
       fill_breed("강아지", "말티즈") == "말티즈")
filled_breed = fill_breed("강아지", None)
report("fill_breed: null → 랜덤 생성",
       filled_breed in DOG_BREEDS,
       f"결과={filled_breed}")
cat_breed = fill_breed("고양이", None)
report("fill_breed: 고양이 랜덤",
       cat_breed in CAT_BREEDS,
       f"결과={cat_breed}")

# fill_age
report("fill_age: 기존 값 유지", fill_age("강아지", 5) == 5)
filled_age = fill_age("강아지", None)
report("fill_age: null → 랜덤 생성",
       filled_age in DOG_AGE_DIST,
       f"결과={filled_age}")

# fill_gender
report("fill_gender: 기존 값 유지", fill_gender("male") == "male")
filled_gender = fill_gender(None)
report("fill_gender: null → 랜덤 생성",
       filled_gender in ("male", "female"),
       f"결과={filled_gender}")

# fill_is_neutered
report("fill_is_neutered: 기존 값 유지",
       fill_is_neutered(3, "male", True) is True)
neutered = fill_is_neutered(3, "male", None)
report("fill_is_neutered: null → 랜덤 생성",
       isinstance(neutered, bool),
       f"결과={neutered}")

# fill_weight
report("fill_weight: 기존 값 유지",
       fill_weight("강아지", "말티즈", 3, "male", 4) == 4)
weight = fill_weight("강아지", "말티즈", 3, "male", None)
report("fill_weight: null → 추정값",
       isinstance(weight, int) and 1 <= weight <= 10,
       f"결과={weight}kg")

cat_weight = fill_weight("고양이", "코리안숏헤어", 2, "female", None)
report("fill_weight: 고양이 체중 추정",
       isinstance(cat_weight, int) and 1 <= cat_weight <= 10,
       f"결과={cat_weight}kg")

large_weight = fill_weight("강아지", "골든리트리버", 3, "male", None)
report("fill_weight: 대형견 체중 추정",
       isinstance(large_weight, int) and 15 <= large_weight <= 50,
       f"결과={large_weight}kg")

# process_scenario
scenario_data = {
    "meta": {
        "article_id": "test_001",
        "is_pet_insurance_related": True,
        "question_intent": "보험사_비교_추천",
    },
    "state": {
        "species": "강아지",
        "breed": None,
        "age": None,
        "gender": None,
        "is_neutered": None,
        "weight": None,
        "health_condition": None,
    },
}

result = process_scenario(scenario_data)
report("process_scenario: 정상 처리", result is not None)
if result:
    state = result["state"]
    report("breed 채워짐", state["breed"] is not None and state["breed"] in DOG_BREEDS,
           f"breed={state['breed']}")
    report("age 채워짐", state["age"] is not None and isinstance(state["age"], int),
           f"age={state['age']}")
    report("gender 채워짐", state["gender"] in ("male", "female"),
           f"gender={state['gender']}")
    report("is_neutered 채워짐", isinstance(state["is_neutered"], bool),
           f"is_neutered={state['is_neutered']}")
    report("weight 채워짐", isinstance(state["weight"], int) and state["weight"] >= 1,
           f"weight={state['weight']}kg")
    report("health_condition 구조 보장",
           isinstance(state["health_condition"], dict),
           f"hc={state['health_condition']}")

# species null → 제외
null_species = {
    "meta": {"is_pet_insurance_related": True},
    "state": {"species": None},
}
report("process_scenario: species null 제외",
       process_scenario(null_species) is None)

# 비관련 글 → 제외
not_related = {
    "meta": {"is_pet_insurance_related": False},
    "state": {"species": "강아지"},
}
report("process_scenario: 비관련 글 제외",
       process_scenario(not_related) is None)


# ================================================================
# 8. extract_to_state (비LLM 부분) 테스트
# ================================================================
print("\n=== 8. extract_to_state (비LLM 부분) 테스트 ===")

if extract_to_state_available:
    # extract_yaml_from_response
    yaml_in_code_block = '```yaml\nmeta:\n  article_id: "123"\n```'
    report("YAML 추출: 코드 블록",
           "article_id" in extract_yaml_from_response(yaml_in_code_block))

    plain_yaml = 'meta:\n  article_id: "123"'
    report("YAML 추출: 일반 텍스트",
           "article_id" in extract_yaml_from_response(plain_yaml))

    # clean_body
    dirty = "앞부분--------실제 질문 내용태그이후생략"
    report("clean_body: 템플릿/태그 제거",
           "실제 질문 내용" in clean_body(dirty))

    # format_comments
    comments = [
        {"author": "전문가", "content": "메리츠 추천합니다"},
        {"author": "일반인", "content": "감사합니다"},
    ]
    formatted = format_comments(comments)
    report("format_comments", "메리츠" in formatted and "전문가" in formatted)

    report("format_comments: 빈 댓글", format_comments([]) == "댓글 없음")

    # validate_and_fix
    raw_data = {
        "meta": {
            "question_intent": "잘못된_의도",
            "user_concerns": ["보험료", "없는항목"],
            "expert_advice": None,
        },
        "pets": [
            {
                "species": "강아지",
                "breed": "말티즈",
                "age": "3",
                "weight": "4.5",
                "preferred_insurers": ["메리츠화재해상보험", "없는보험사"],
                "health_condition": None,
            }
        ],
    }
    article_for_fix = {"article_id": "99999", "written_date": "2024.01.01. 12:00"}
    fixed = validate_and_fix(raw_data, article_for_fix)

    report("validate_and_fix: article_id 보정",
           fixed["meta"]["article_id"] == "99999")
    report("validate_and_fix: 잘못된 intent → 일반_문의",
           fixed["meta"]["question_intent"] == "일반_문의")
    report("validate_and_fix: 유효하지 않은 concern 제거",
           "없는항목" not in fixed["meta"]["user_concerns"])
    report("validate_and_fix: expert_advice 기본값",
           isinstance(fixed["meta"]["expert_advice"], dict))
    report("validate_and_fix: age 정수 변환",
           fixed["pets"][0]["age"] == 3)
    report("validate_and_fix: weight 정수 변환",
           fixed["pets"][0]["weight"] == 4)
    report("validate_and_fix: 없는 보험사 제거",
           "없는보험사" not in (fixed["pets"][0]["preferred_insurers"] or []))
    report("validate_and_fix: health_condition 구조 보장",
           isinstance(fixed["pets"][0]["health_condition"], dict))
else:
    print("  (extract_to_state 모듈 미로드 — UPSTAGE_API_KEY 없음, 건너뜀)")


# ================================================================
# 결과 요약
# ================================================================
print(f"\n{'='*60}")
print(f"테스트 결과: {PASS} PASS / {FAIL} FAIL (총 {PASS + FAIL}건)")
if ERRORS:
    print(f"\n실패 목록:")
    for e in ERRORS:
        print(e)
print(f"{'='*60}")

if __name__ == "__main__":
    sys.exit(1 if FAIL > 0 else 0)
