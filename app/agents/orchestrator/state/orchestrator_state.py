from app.agents.judge_agent.state import JudgeAgentState
from app.agents.vet_agent.state import VetAgentState
from app.agents.rag_agent.state.rag_state import RagState


class OrchestratorState(JudgeAgentState, RagState, VetAgentState ): ...
