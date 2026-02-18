"""
네이버 YAML 데이터 기반 테스트셋 로더.

지정된 폴더(app/agents/user_input_template_agent/samples/naver)에 있는
약 128개의 YAML 파일을 읽어 VetAgentState 객체로 변환합니다.

YAML 구조 예시:
    meta:
      article_id: '34840'
      ...
    state:
      species: 강아지
      breed: 토이푸들
      age: 3
      health_condition:
        frequent_illness_area: 슬개구
        disease_surgery_history: 감기, 피부병(2년 전), 슬개골 1기 진단
      ...
"""

import logging
from pathlib import Path

import yaml
from pydantic import ValidationError

from app.agents.user_input_template_agent.state import HealthCondition
from app.agents.vet_agent.state import VetAgentState

logger = logging.getLogger(__name__)

# 네이버 샘플 YAML 기본 경로
DEFAULT_SAMPLE_DIR = (
    Path(__file__).parents[1]
    / "agents"
    / "user_input_template_agent"
    / "samples"
    / "naver"
)


def _parse_health_condition(raw: dict | None) -> HealthCondition | None:
    """health_condition 딕셔너리를 안전하게 HealthCondition 객체로 변환합니다.

    일부 YAML에서 frequent_illness_area나 disease_surgery_history가
    null일 수 있으므로 dict.get()으로 안전하게 처리합니다.

    Args:
        raw: YAML에서 파싱된 health_condition 딕셔너리 (None일 수 있음)

    Returns:
        HealthCondition 객체 또는 None
    """
    if raw is None or not isinstance(raw, dict):
        return None

    return HealthCondition(
        frequent_illness_area=raw.get("frequent_illness_area"),
        disease_surgery_history=raw.get("disease_surgery_history"),
    )


def load_single_yaml(yaml_path: Path) -> tuple[str, VetAgentState]:
    """단일 YAML 파일을 읽어 (파일명, VetAgentState) 튜플을 반환합니다.

    Args:
        yaml_path: YAML 파일의 절대/상대 경로

    Returns:
        (파일명(확장자 제외), VetAgentState) 튜플

    Raises:
        ValidationError: YAML 데이터가 VetAgentState 스키마에 맞지 않을 때
        FileNotFoundError: YAML 파일이 존재하지 않을 때
    """
    data = yaml.safe_load(yaml_path.read_text(encoding="utf-8"))
    state_dict = data.get("state", {})

    # health_condition 안전 파싱
    # YAML에서 health_condition이 null이거나 하위 필드가 null인 경우를 처리
    state_dict["health_condition"] = _parse_health_condition(
        state_dict.get("health_condition")
    )

    state = VetAgentState.model_validate(state_dict)
    return yaml_path.stem, state


def load_all_yaml_states(
    sample_dir: Path = DEFAULT_SAMPLE_DIR,
    limit: int | None = None,
) -> list[tuple[str, VetAgentState]]:
    """지정 폴더의 모든 YAML 파일을 읽어 VetAgentState 리스트로 반환합니다.

    Args:
        sample_dir: YAML 파일들이 들어있는 디렉토리 경로
        limit: 최대 로드 개수 (None이면 전체, API 비용 절감 시 3 등으로 설정)

    Returns:
        [(파일명, VetAgentState), ...] 리스트 (파일명 기준 정렬)
    """
    yaml_files = sorted(sample_dir.glob("*.yaml"))

    if limit is not None:
        yaml_files = yaml_files[:limit]

    results: list[tuple[str, VetAgentState]] = []

    for yaml_path in yaml_files:
        try:
            file_name, state = load_single_yaml(yaml_path)
            results.append((file_name, state))
        except ValidationError as exc:
            # 스키마 불일치 파일은 경고 후 건너뛰기 (배치 처리 안정성)
            logger.warning("YAML 파싱 실패 [%s]: %s", yaml_path.name, exc)

    logger.info(
        "총 %d/%d개 YAML 파일 로드 완료 (디렉토리: %s)",
        len(results),
        len(yaml_files),
        sample_dir,
    )
    return results


if __name__ == "__main__":
    # 단독 실행 시 로드 테스트: uv run python -m app.evaluation.data_loader
    from rich import print as rprint

    logging.basicConfig(level=logging.INFO)

    states = load_all_yaml_states(limit=3)
    for name, s in states:
        rprint(f"\n[bold]{name}[/bold]")
        rprint(f"  종: {s.species}, 품종: {s.breed}, 나이: {s.age}")

        # health_condition 내부 필드 출력 (null-safe)
        hc = s.health_condition
        if hc:
            rprint(f"  자주 아픈 부위: {hc.frequent_illness_area or '없음'}")
            rprint(f"  질병/수술 이력: {hc.disease_surgery_history or '없음'}")
        else:
            rprint("  건강 상태 정보: 없음")
