# pet-insurance-recommender

Multi-agent project structure for a FastAPI-served insurance recommendation system.

## Structure

- `app/main.py`: FastAPI application entrypoint
- `app/agents/orchestrator/graph.py`: top-level routing/orchestration graph
- `app/agents/sample_agent_1/`: agent 1 implementation (`graph.py`, `nodes/`, `state/`, `tools/`)
- `app/agents/sample_agent_2/`: agent 2 implementation (`graph.py`, `nodes/`, `state/`, `tools/`)

## Run

### 특정 .py 파일의 main 함수 실행
```bash
# e.g. app/agents/hellow_world.py
uv run python -m app.agents.hellow_world
```

### Qdrant

```bash
docker compose -f docker-compose.qdrant.yml up -d
```

### FastAPI

```bash
uv run uvicorn app.main:app --reload
```
