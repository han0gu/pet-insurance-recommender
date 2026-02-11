from app.agents.vet_agent.state import VetAgentState
from app.agents.rag_agent.state.rag_state import RagState


class OrchestratorState(VetAgentState, RagState): ...
