from enum import Enum
from typing import Optional
from pydantic import BaseModel, Field


# 성별 Enum
class Gender(str, Enum):
    male = "male"
    female = "female"


# 보장 스타일 Enum
class CoverageStyle(str, Enum):
    minimal = "minimal"
    comprehensive = "comprehensive"


# 건강 상태
class HealthCondition(BaseModel):
    frequent_illness_area: Optional[str] = Field(
        default=None, description="자주 아픈 부위(예: 머리, 발, 등)"
    )
    disease_surgery_history: Optional[str] = Field(
        default=None, description="질병/수술 이력(예: 간염, 당뇨, 수술 이력 없음)"
    )


# 펫보험 상품 추천을 위한 고객 입력 템플릿 State
class UserInputTemplateState(BaseModel):
    species: Optional[str] = Field(default=None, description="종")
    breed: Optional[str] = Field(default=None, description="품종")
    age: Optional[int] = Field(default=None, description="나이")
    gender: Optional[Gender] = Field(default=None, description="성별")
    is_neutered: Optional[bool] = Field(default=None, description="중성화 여부")
    weight: Optional[int] = Field(default=None, description="체중(kg)")
    health_condition: Optional[HealthCondition] = Field(
        default=None, description="건강 상태"
    )
    coverage_style: Optional[CoverageStyle] = Field(
        default=None, description="보장 스타일"
    )


if __name__ == "__main__":
    from rich import print as rprint
    from app.agents.user_input_template_agent.utils.cli import (
        create_arg_parser,
        load_state_from_yaml,
    )

    args = create_arg_parser().parse_args()
    state = load_state_from_yaml(args.input, UserInputTemplateState)
    rprint(state)
