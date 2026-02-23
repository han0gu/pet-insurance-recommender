#!/usr/bin/env python3
"""
data/interim/scenarios_v2_yaml의 null 필수 필드를 대한민국 반려동물 통계 기반으로 채워
data/processed/scenarios_v3_yaml에 저장하는 스크립트.

처리 규칙:
  species  → null이면 데이터 제외
  breed    → 대한민국 인기 품종 분포 기반 랜덤
  age      → 대한민국 반려동물 연령 분포 기반 랜덤
  gender   → 대한민국 반려동물 성비 기반 랜덤 (male 53%, female 47%)
  is_neutered → age, gender 기반 조건부 확률 추정
  weight   → species, breed, age, gender 기반 추정

사용법:
  python fill_required_fields.py
  python fill_required_fields.py --seed 42  # 재현성을 위한 시드 지정
"""

import argparse
import random
import yaml
from pathlib import Path


# ============================================================
# 대한민국 반려동물 통계 기반 분포 테이블
# 출처: 농림축산식품부 동물보호 관리시스템, KB경영연구소 반려동물 보고서 (2024)
# ============================================================

# --- 품종 분포 (가중치) ---
DOG_BREEDS = {
    "말티즈":         20,
    "토이푸들":       13,
    "말티푸":         10,
    "비숑프리제":      8,
    "포메라니안":      7,
    "시츄":            5,
    "치와와":          4,
    "요크셔테리어":    3,
    "코카스파니엘":    2,
    "미니어처슈나우저": 2,
    "닥스훈트":        2,
    "진돗개":          3,
    "골든리트리버":    2,
    "웰시코기":        2,
    "래브라도리트리버": 1,
    "시바이누":        2,
    "비글":            1,
    "믹스견":         13,
}

CAT_BREEDS = {
    "코리안숏헤어":    38,
    "러시안블루":      10,
    "브리티시숏헤어":  10,
    "스코티시폴드":     8,
    "페르시안":         5,
    "아메리칸숏헤어":   5,
    "랙돌":             6,
    "뱅갈":             3,
    "먼치킨":           4,
    "터키시앙고라":     3,
    "노르웨이숲":       2,
    "샴":               2,
    "믹스묘":           4,
}

# --- 연령 분포 (세, 가중치) ---
# 펫보험 가입 대상 연령대 반영 (0세~10세)
DOG_AGE_DIST = {
    0: 5,   # 0세 (3개월~11개월)
    1: 18,
    2: 16,
    3: 15,
    4: 12,
    5: 10,
    6: 8,
    7: 6,
    8: 5,
    9: 3,
    10: 2,
}

CAT_AGE_DIST = {
    0: 5,
    1: 16,
    2: 15,
    3: 14,
    4: 12,
    5: 10,
    6: 8,
    7: 7,
    8: 5,
    9: 4,
    10: 4,
}

# --- 성별 분포 ---
# 대한민국 반려동물 성비: male 약 53%, female 약 47%
GENDER_DIST = {"male": 53, "female": 47}

# --- 중성화 조건부 확률 (age, gender → 중성화 확률 %) ---
# 한국 반려동물 중성화율: 전체 약 63% (2024 기준)
# 나이가 어릴수록 미중성화, 암컷이 중성화 비율 약간 높음
NEUTERED_PROB = {
    # (age_range, gender) → 중성화 확률 %
    "young_male":   35,   # 0~1세, 수컷
    "young_female": 45,   # 0~1세, 암컷
    "mid_male":     65,   # 2~4세, 수컷
    "mid_female":   75,   # 2~4세, 암컷
    "old_male":     75,   # 5세 이상, 수컷
    "old_female":   85,   # 5세 이상, 암컷
}

# --- 품종별 성견/성묘 평균 체중 (kg) ---
DOG_WEIGHT = {
    "말티즈":         (2.5, 3.5),
    "토이푸들":       (2.5, 4.0),
    "말티푸":         (2.5, 5.0),
    "비숑프리제":     (3.0, 5.5),
    "포메라니안":     (1.8, 3.2),
    "시츄":           (4.0, 7.5),
    "치와와":         (1.5, 3.0),
    "요크셔테리어":   (1.5, 3.2),
    "코카스파니엘":   (11.0, 14.0),
    "미니어처슈나우저": (5.0, 8.0),
    "닥스훈트":       (4.0, 5.5),
    "진돗개":         (18.0, 25.0),
    "골든리트리버":   (25.0, 34.0),
    "웰시코기":       (10.0, 14.0),
    "래브라도리트리버": (25.0, 36.0),
    "시바이누":       (8.0, 11.0),
    "비글":           (9.0, 11.0),
    "믹스견":         (5.0, 15.0),
}

CAT_WEIGHT = {
    "코리안숏헤어":    (3.5, 5.5),
    "러시안블루":      (3.0, 5.5),
    "브리티시숏헤어":  (4.0, 7.0),
    "스코티시폴드":    (3.0, 5.5),
    "페르시안":        (3.0, 5.5),
    "아메리칸숏헤어":  (3.5, 6.0),
    "랙돌":            (4.5, 7.0),
    "뱅갈":            (3.5, 6.0),
    "먼치킨":          (2.5, 4.0),
    "터키시앙고라":    (3.0, 5.0),
    "노르웨이숲":      (4.5, 8.0),
    "샴":              (3.0, 5.0),
    "믹스묘":          (3.5, 5.5),
}


# ============================================================
# 랜덤 생성 함수
# ============================================================
def weighted_choice(dist: dict):
    """가중치 딕셔너리에서 랜덤 선택"""
    items = list(dist.keys())
    weights = list(dist.values())
    return random.choices(items, weights=weights, k=1)[0]


def fill_breed(species: str, current_breed) -> str:
    """품종 채우기: null이면 통계 기반 랜덤 생성"""
    if current_breed is not None:
        return current_breed
    if species == "강아지":
        return weighted_choice(DOG_BREEDS)
    elif species == "고양이":
        return weighted_choice(CAT_BREEDS)
    return None


def fill_age(species: str, current_age) -> int:
    """나이 채우기: null이면 통계 기반 랜덤 생성"""
    if current_age is not None:
        return int(float(current_age))
    if species == "강아지":
        return weighted_choice(DOG_AGE_DIST)
    elif species == "고양이":
        return weighted_choice(CAT_AGE_DIST)
    return None


def fill_gender(current_gender) -> str:
    """성별 채우기: null이면 통계 기반 랜덤 생성"""
    if current_gender is not None:
        return current_gender
    return weighted_choice(GENDER_DIST)


def fill_is_neutered(age: int, gender: str, current_neutered) -> bool:
    """중성화 여부 추정: age + gender 기반 조건부 확률"""
    if current_neutered is not None:
        return current_neutered

    if age is None or gender is None:
        # 정보 부족 시 전체 평균 (63%)
        return random.random() < 0.63

    # 나이 구간 결정
    if age <= 1:
        age_group = "young"
    elif age <= 4:
        age_group = "mid"
    else:
        age_group = "old"

    key = f"{age_group}_{gender}"
    prob = NEUTERED_PROB.get(key, 63)
    return random.random() < (prob / 100)


def fill_weight(species: str, breed: str, age: int, gender: str, current_weight) -> int:
    """체중 추정: species + breed + age + gender 기반"""
    if current_weight is not None:
        return int(float(current_weight))

    # 품종별 성체 체중 범위 조회
    if species == "강아지":
        weight_range = DOG_WEIGHT.get(breed, (4.0, 10.0))
    elif species == "고양이":
        weight_range = CAT_WEIGHT.get(breed, (3.5, 5.5))
    else:
        return None

    low, high = weight_range

    # 성별 보정: 수컷이 약 10% 무거움
    if gender == "male":
        mid = (low + high) / 2 * 1.05
    else:
        mid = (low + high) / 2 * 0.95

    # 나이 보정: 어린 개체는 성체 대비 비율 적용
    if age is not None:
        if age == 0:
            ratio = 0.55  # 3~11개월: 성체의 55%
        elif age == 1:
            ratio = 0.85  # 1세: 성체의 85%
        else:
            ratio = 1.0   # 2세 이상: 성체
    else:
        ratio = 1.0

    estimated = mid * ratio

    # 표준편차 적용한 약간의 랜덤성
    spread = (high - low) * 0.15
    weight = random.gauss(estimated, spread)

    # 최소 1kg, 정수 반올림
    return max(1, round(weight))


# ============================================================
# 메인 처리
# ============================================================
def process_scenario(data: dict) -> dict | None:
    """시나리오의 null 필수 필드를 채움. species가 null이면 None 반환 (제외)."""
    meta = data.get("meta", {})
    state = data.get("state", {})

    # species null → 제외
    species = state.get("species")
    if species is None or species == "null":
        return None

    # 펫보험 비관련 → 그대로 유지 (제외하지 않고 보존)
    if not meta.get("is_pet_insurance_related", False):
        return None

    # 순서대로 채우기 (후속 필드가 이전 필드에 의존)
    state["breed"] = fill_breed(species, state.get("breed"))
    state["age"] = fill_age(species, state.get("age"))
    state["gender"] = fill_gender(state.get("gender"))
    state["is_neutered"] = fill_is_neutered(
        state.get("age"), state.get("gender"), state.get("is_neutered")
    )
    state["weight"] = fill_weight(
        species,
        state.get("breed"),
        state.get("age"),
        state.get("gender"),
        state.get("weight"),
    )

    # health_condition 기본 구조 보장
    hc = state.get("health_condition")
    if not isinstance(hc, dict):
        state["health_condition"] = {
            "frequent_illness_area": None,
            "disease_surgery_history": None,
        }

    data["meta"] = meta
    data["state"] = state
    return data


def main():
    parser = argparse.ArgumentParser(
        description="필수 필드 null 값을 대한민국 반려동물 통계 기반으로 채움"
    )
    parser.add_argument(
        "--seed", type=int, default=42, help="랜덤 시드 (기본: 42)"
    )
    parser.add_argument(
        "--input-dir", type=str, default="data/interim/scenarios_v2_yaml", help="입력 디렉토리"
    )
    parser.add_argument(
        "--output-dir", type=str, default="data/processed/scenarios_v3_yaml", help="출력 디렉토리"
    )
    args = parser.parse_args()

    random.seed(args.seed)

    project_root = Path(__file__).resolve().parent.parent
    input_dir = project_root / args.input_dir
    output_dir = project_root / args.output_dir
    output_dir.mkdir(exist_ok=True)

    input_files = sorted(input_dir.glob("*.yaml"))
    total = len(input_files)

    stats = {
        "included": 0,
        "excluded_no_species": 0,
        "excluded_not_related": 0,
        "breed_filled": 0,
        "age_filled": 0,
        "gender_filled": 0,
        "neutered_filled": 0,
        "weight_filled": 0,
    }

    for f in input_files:
        with open(f, "r", encoding="utf-8") as fp:
            data = yaml.safe_load(fp)

        state = data.get("state", {})
        meta = data.get("meta", {})

        # 비관련 게시글 제외
        if not meta.get("is_pet_insurance_related", False):
            stats["excluded_not_related"] += 1
            continue

        # 기존 null 여부 기록 (통계용)
        was_null = {
            "breed": state.get("breed") is None,
            "age": state.get("age") is None,
            "gender": state.get("gender") is None,
            "neutered": state.get("is_neutered") is None,
            "weight": state.get("weight") is None,
        }

        result = process_scenario(data)
        if result is None:
            stats["excluded_no_species"] += 1
            continue

        # 통계 갱신
        stats["included"] += 1
        if was_null["breed"]:
            stats["breed_filled"] += 1
        if was_null["age"]:
            stats["age_filled"] += 1
        if was_null["gender"]:
            stats["gender_filled"] += 1
        if was_null["neutered"]:
            stats["neutered_filled"] += 1
        if was_null["weight"]:
            stats["weight_filled"] += 1

        # 저장
        out_path = output_dir / f.name
        with open(out_path, "w", encoding="utf-8") as fp:
            yaml.dump(
                result,
                fp,
                allow_unicode=True,
                default_flow_style=False,
                sort_keys=False,
            )

    # 결과 출력
    print(f"입력 파일:           {total}개")
    print(f"출력 (포함):         {stats['included']}개")
    print(f"제외 (species null): {stats['excluded_no_species']}개")
    print(f"제외 (비관련):       {stats['excluded_not_related']}개")
    print()
    print(f"=== 채운 필드 수 ===")
    print(f"  breed:       {stats['breed_filled']}건")
    print(f"  age:         {stats['age_filled']}건")
    print(f"  gender:      {stats['gender_filled']}건")
    print(f"  is_neutered: {stats['neutered_filled']}건")
    print(f"  weight:      {stats['weight_filled']}건")
    print(f"\n출력 위치: {output_dir}/")


if __name__ == "__main__":
    main()
