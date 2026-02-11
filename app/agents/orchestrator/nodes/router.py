from typing import Literal

from app.agents.orchestrator.state.orchestrator_state import OrchestratorState


def route_after_user_input(
    state: OrchestratorState,
) -> Literal["vet_diagnosis", "RAG"]:
    """diseases 존재 여부에 따라 다음 노드 결정"""
    if state.diseases:
        print("  [Router] diseases 존재 → RAG로 이동")
        return "RAG"
    else:
        print("  [Router] diseases 없음 → vet_diagnosis로 이동")
        return "vet_diagnosis"
