from pydantic import BaseModel, Field

from user_input_template_agent.state import UserInputTemplateState


# 질병 정보
class DiseaseInfo(BaseModel):
    name: str = Field(description="질병명")
    incidence_rate: str = Field(description="발병률")
    onset_period: str = Field(description="발병시기")


# 수의사 에이전트 출력 State
class VetAgentOutputState(BaseModel):
    diseases: list[DiseaseInfo] = Field(default_factory=list, description="질병 목록")


# 수의사 에이전트 전체 State (입력 + 출력)
class VetAgentState(UserInputTemplateState):
    diseases: list[DiseaseInfo] = Field(default_factory=list, description="질병 목록")


if __name__ == "__main__":
    from rich import print as rprint
    from vet_agent.utils.cli import create_arg_parser, load_state_from_yaml

    args = create_arg_parser().parse_args()
    state = load_state_from_yaml(args.input, VetAgentState)
    rprint(state)
