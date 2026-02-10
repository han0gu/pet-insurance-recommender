from user_input_template_agent.state import UserInputTemplateState


def user_input_template_node(
    state: UserInputTemplateState,
) -> UserInputTemplateState:
    """
    펫보험 상품 추천을 위한 고객 입력 템플릿 노드
    """
    return state


if __name__ == "__main__":
    from rich import print as rprint
    from user_input_template_agent.state import UserInputTemplateState
    from user_input_template_agent.utils.cli import create_arg_parser, load_state_from_yaml

    args = create_arg_parser().parse_args()
    state = load_state_from_yaml(args.input, UserInputTemplateState)
    result = user_input_template_node(state)
    rprint(result)
