from langgraph.graph import StateGraph, START, END
from user_input_template_agent.nodes import user_input_template_node
from user_input_template_agent.state import UserInputTemplateState


builder = StateGraph(UserInputTemplateState)

builder.add_node("user_input_template", user_input_template_node)
builder.add_edge(START, "user_input_template")
builder.add_edge("user_input_template", END)

graph = builder.compile()

if __name__ == "__main__":
    from rich import print as rprint
    from user_input_template_agent.utils.cli import create_arg_parser, load_state_from_yaml

    args = create_arg_parser().parse_args()
    state = load_state_from_yaml(args.input, UserInputTemplateState)
    result = graph.invoke(state)
    rprint(result)
