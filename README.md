# pet-insurance-recommender

Multi-agent project structure for a FastAPI-served insurance recommendation system.

## Structure

- `app/main.py`: FastAPI application entrypoint
- `app/agents/orchestrator/`: 최상위 오케스트레이션 그래프 (`orchestrator_graph.py`, `nodes/`, `state/`)
- `app/agents/user_input_template_agent/`: 사용자 입력 처리 (`graph.py`, `nodes/`, `state/`, `middleware/`, `samples/`)
- `app/agents/vet_agent/`: 수의사 진단 에이전트 (`graph.py`, `nodes/`, `state/`)
- `app/agents/rag_agent/`: 보험 약관 검색 에이전트 (`rag_graph.py`, `nodes/`, `state/`, `tools/`)
- `app/agents/judge_agent/`: 보험 상품 검증 에이전트 (`graph.py`, `nodes/`, `state/`)
- `app/agents/composer_agent/`: 최종 답변 생성 에이전트 (`graph.py`, `nodes/`)
- `app/agents/document_parser/`: 약관 PDF 파싱/임베딩/적재 파이프라인

## Run

### Orchestrator Graph

```bash
# 기본 실행 (default YAML 입력)
bash script/run_orchestrator_graph.sh

# YAML 파일 및 thread_id 지정
bash script/run_orchestrator_graph_args.sh
```

### Orchestrator Test

```bash
# 라우터 분기 테스트 (동일 thread_id로 2회 순차 실행, 체크포인터 상태 유지 검증)
bash script/run_test_router.sh

# 단일 입력 파이프라인 테스트
bash script/run_test_single.sh
```

### Guardrail Test

```bash
# 입력 Sanitization 가드레일 테스트
# - 정상 입력 회귀 테스트 + 프롬프트 인젝션 정화 테스트
bash script/run_test_guardrail.sh
```

### Qdrant

```bash
docker compose -f docker-compose.qdrant.yml up -d
```

### Streamlit Demo

```bash
bash script/run_streamlit.sh
```

### FastAPI

```bash
uv run uvicorn app.main:app --reload
```

## 기타

### 약관 파일 처리

#### 1. 파일 준비
- 경로: `app/agents/document_parser/data/terms/`
- 예시: `meritz_1_maum_pet_12_61.pdf`

#### 2. Qdrant 실행
```bash
docker compose -f docker-compose.qdrant.yml up -d
```

#### 3. 파싱/청킹 및 적재(optional)
```bash
# 파싱, 청킹만
uv run python -m app.agents.document_parser.dp_graph --file-name meritz_1_maum_pet_12_61.pdf

# DB 적재까지
uv run python -m app.agents.document_parser.dp_graph --file-name meritz_1_maum_pet_12_61.pdf --ingest
```

#### 4. Qdrant dashboard 확인
```bash
http://localhost:6333/dashboard
```
