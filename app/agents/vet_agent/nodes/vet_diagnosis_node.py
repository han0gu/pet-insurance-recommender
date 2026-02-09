import json

from vet_agent.model.model import llm
from vet_agent.state import VetAgentOutputState, VetAgentState


def vet_diagnosis_node(state: VetAgentState) -> dict:
    """반려동물 취약 질병 정보를 LLM으로 분석하는 노드"""
    structured_llm = llm.with_structured_output(VetAgentOutputState)
    input_summary = json.dumps(
        state.model_dump(exclude={"diseases"}, exclude_none=True),
        ensure_ascii=False,
        indent=2,
    )
    result = structured_llm.invoke(
        f"""당신은 수의사입니다.
다음 반려동물 정보를 기반으로 해당 반려동물이 잘 걸리는 질병을 분석해주세요.
각 질병에 대해 질병명, 발병률, 발병시기를 포함해주세요.

반려동물 정보:
{input_summary}"""
    )
    return {"diseases": result.diseases}


if __name__ == "__main__":
    from rich import print as rprint
    from vet_agent.utils.cli import create_arg_parser, load_state_from_yaml

    args = create_arg_parser().parse_args()
    state = load_state_from_yaml(args.input, VetAgentState)
    result = vet_diagnosis_node(state)
    rprint(result)
