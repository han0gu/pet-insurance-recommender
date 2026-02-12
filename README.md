# pet-insurance-recommender

Multi-agent project structure for a FastAPI-served insurance recommendation system.

## Structure

- `app/main.py`: FastAPI application entrypoint
- `app/agents/orchestrator/graph.py`: top-level routing/orchestration graph
- `app/agents/sample_agent_1/`: agent 1 implementation (`graph.py`, `nodes/`, `state/`, `tools/`)
- `app/agents/sample_agent_2/`: agent 2 implementation (`graph.py`, `nodes/`, `state/`, `tools/`)

## Run

### Qdrant

```bash
docker compose -f docker-compose.qdrant.yml up -d
```

### FastAPI

```bash
uv run uvicorn app.main:app --reload
```

## 기타

### 약관 파일 처리

#### 1. 파일 준비
- 경로: `app/agents/document_parser/data/terms/`
- 예시: `meritz_maum_pet_12_61.pdf`

#### 2. Qdrant 실행
```bash
docker compose -f docker-compose.qdrant.yml up -d
```

#### 3. 파싱/청킹 및 적재(optional)
```bash
# 파싱, 청킹만
uv run python -m app.agents.document_parser.dp_graph --file-name meritz_maum_pet_12_61.pdf

# DB 적재까지
uv run python -m app.agents.document_parser.dp_graph --file-name meritz_maum_pet_12_61.pdf --ingest
```

#### 4. Qdrant dashboard 확인
```bash
http://localhost:6333/dashboard
```
