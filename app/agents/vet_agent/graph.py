from langgraph.graph import END, START, StateGraph

from app.agents.user_input_template_agent.state import UserInputTemplateState
from app.agents.vet_agent.nodes import vet_diagnosis_node
from app.agents.vet_agent.state import VetAgentOutputState, VetAgentState

builder = StateGraph(
    VetAgentState,
    input_schema=UserInputTemplateState,
    output_schema=VetAgentOutputState,
)

builder.add_node("vet_diagnosis", vet_diagnosis_node)
builder.add_edge(START, "vet_diagnosis")
builder.add_edge("vet_diagnosis", END)

graph = builder.compile()

if __name__ == "__main__":
    from rich import print as rprint
    from app.agents.vet_agent.utils.cli import create_arg_parser, load_state_from_yaml

    args = create_arg_parser().parse_args()
    state = load_state_from_yaml(args.input, UserInputTemplateState)
    result = graph.invoke(state)
    rprint(result)
