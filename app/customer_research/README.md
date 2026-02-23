# Customer Research

네이버 카페 게시글을 수집하고, 펫보험 관련 질문을 구조화된 시나리오 데이터로 변환하는 파이프라인 모듈이다.

## 목차

- [디렉토리 구조](#디렉토리-구조)
- [데이터 흐름](#데이터-흐름)
- [모듈 설명](#모듈-설명)
  - [feature](#feature)
  - [pipeline](#pipeline)
- [데이터 명세](#데이터-명세)
- [사전 요구사항](#사전-요구사항)
- [사용 방법](#사용-방법)
- [참조목록](#참조목록)

---

## 디렉토리 구조

```
app/customer_research/
├── feature/
│   ├── http_client.py            # HTTP 요청 유틸리티
│   ├── naver_cafe_parser.py      # 카페 목록 페이지 HTML 파싱
│   ├── naver_article_parser.py   # 카페 게시글 상세 HTML 파싱
│   └── naver_crawler.py          # 카페 게시판 크롤링 (목록 수집)
└── pipeline/
    ├── __init__.py
    ├── extract_scenarios.py      # Stage 1: 규칙 기반 시나리오 추출
    ├── summarize_with_llm.py     # Stage 2: LLM 기반 질문 요약
    ├── extract_to_state.py       # Stage 3: LLM 기반 구조화 (YAML)
    └── fill_required_fields.py   # Stage 4: 통계 기반 필드 보정
```

---

## 데이터 흐름

```
네이버 카페 URL
  │
  ▼
[naver_crawler] ── 게시판 목록 페이지 순회
  │
  ▼
[naver_article_parser] ── 개별 게시글 HTML 수집 및 파싱
  │
  ▼
data/raw/articles/*.json          ← 원본 게시글 (제목, 본문, 댓글, 첨부 등)
  │
  ├─▶ [extract_scenarios]         ← Stage 1: 규칙 기반 필터 + 시나리오 생성
  │     │
  │     ▼
  │   data/interim/scenarios/     ← SC_{id}.json, scenarios_summary.json
  │     │
  │     ▼
  │   [summarize_with_llm]        ← Stage 2: LLM으로 질문 요약 보강
  │
  └─▶ [extract_to_state]          ← Stage 3: LLM으로 meta + pets YAML 추출
        │
        ▼
      data/interim/scenarios_v2_yaml/*.yaml
        │
        ▼
      [fill_required_fields]      ← Stage 4: null 필드를 통계 기반으로 채움
        │
        ▼
      data/processed/scenarios_v3_yaml/*.yaml  ← 최종 시나리오 데이터
```

---

## 모듈 설명

### feature

데이터 수집 계층으로, 네이버 카페의 게시판 목록과 개별 게시글을 크롤링한다.

| 파일 | 역할 |
|------|------|
| `http_client.py` | `requests` 기반 HTTP GET 유틸리티. `fetch_html`(예외 발생)과 `fetch_html_safe`(에러 반환) 두 가지 인터페이스 제공 |
| `naver_cafe_parser.py` | 게시판 목록 HTML에서 `(글번호, 제목, 링크)` 리스트를 추출하고 공지글을 필터링. 다음 페이지 URL도 탐지 |
| `naver_article_parser.py` | 게시글 상세 페이지 HTML에서 제목, 본문, 댓글, 첨부이미지, 작성일, 게시판명을 파싱. Playwright 브라우저 또는 HTTP 폴백으로 HTML 수집 |
| `naver_crawler.py` | 게시판 URL을 입력받아 여러 페이지를 순회하며 게시글 제목/링크 목록을 수집. Playwright 또는 HTTP + 파서 조합으로 동작 |

### pipeline

수집된 게시글 데이터를 시나리오로 변환하는 4단계 파이프라인이다.

#### Stage 1 - `extract_scenarios.py`

규칙 기반으로 펫보험 관련 게시글을 필터링하고 시나리오를 생성한다.

- 키워드 매칭으로 펫보험 질문 여부 판별 (`is_pet_insurance_question`)
- 본문에서 반려동물 정보(종, 품종, 나이, 체중, 특이사항) 추출
- 사용자 관심사(보험료, 보장범위, 슬개골 등)와 관심 보험사 추출
- 질문 의도 분류: 가입 적합성, 보험사 비교, 보장내용, 보험료, 보험금 청구, 일반 문의

#### Stage 2 - `summarize_with_llm.py`

Upstage Solar-pro LLM을 사용하여 게시글 본문을 150자 이내로 요약한다. Stage 1에서 생성된 시나리오의 `original_question_summary` 필드를 보강한다.

#### Stage 3 - `extract_to_state.py`

Upstage Solar-pro LLM으로 게시글+댓글에서 구조화된 데이터를 추출한다.

- `meta`: 질문 의도, 관심사, 설계사 답변 여부 등
- `pets`: 반려동물별 state (종, 품종, 나이, 성별, 중성화, 체중, 건강상태, 보장 스타일, 선호 보험사)
- YAML 스키마로 출력하며, 검증 및 정규화 후 저장
- 한 게시글에 여러 반려동물이 언급된 경우 별도 YAML 파일로 분리

#### Stage 4 - `fill_required_fields.py`

대한민국 반려동물 통계를 기반으로 null인 필드를 확률적으로 채운다.

- 품종: 종별 인기 품종 분포 (예: 말티즈 24%, 푸들 15% 등)
- 나이: 종별 연령 분포
- 성별: 수컷 53%, 암컷 47%
- 중성화: 나이/성별 기반 조건부 확률
- 체중: 품종별 표준 범위 + 성별/나이 보정 + 가우시안 노이즈

---

## 데이터 명세

### 원본 게시글 (`data/raw/articles/*.json`)

```yaml
article_id: "64944"
title: "강아지 펫보험 추천해주세요"
body: "..."
attachments: [{src: "...", alt: "..."}]
comments: [{author: "...", content: "..."}]
written_date: "2024.01.15. 14:30"
page_url: "https://..."
board: "펫보험 Q&A"
```

### 시나리오 v2 (`data/interim/scenarios_v2_yaml/*.yaml`)

```yaml
meta:
  article_id: "64944"
  written_date: "2024.01.15. 14:30"
  is_pet_insurance_related: true
  question_intent: "보험사_비교"
  user_concerns: ["보험료", "보장범위"]
  original_question_summary: "..."
  total_pets: 1
  expert_advice: null
state:
  species: "dog"
  breed: "말티즈"
  age: 3
  gender: "male"
  is_neutered: true
  weight: 4
  health_condition:
    has_disease: false
    disease_name: null
    is_surgery: false
    surgery_name: null
  coverage_style: "실속형"
  preferred_insurers: ["메리츠화재", "DB손해보험"]
```

### 최종 시나리오 v3 (`data/processed/scenarios_v3_yaml/*.yaml`)

v2와 동일한 스키마이며, null이었던 필드가 통계 기반 값으로 채워진 버전이다.

---

## 사전 요구사항

- Python 3.10+
- `.env` 파일에 `UPSTAGE_API_KEY` 설정 (Stage 2, 3에서 사용)
- Playwright 설치 (브라우저 기반 크롤링 시 필요)

```bash
pip install playwright
playwright install
```

### 주요 의존성

| 패키지 | 용도 |
|--------|------|
| `requests` | HTTP 요청 |
| `beautifulsoup4` | HTML 파싱 |
| `playwright` | 브라우저 자동화 (iframe 렌더링이 필요한 네이버 카페) |
| `langchain-upstage` | Upstage Solar-pro LLM 호출 |
| `langchain-core` | LangChain 메시지 구조 |
| `python-dotenv` | 환경변수 로드 |
| `pyyaml` | YAML 입출력 |

---

## 사용 방법

### 게시글 수집

`script/` 디렉토리에 실행 스크립트가 준비되어 있다.

**1단계: 게시판 목록 크롤링** (`script/run_crawler.sh`)

게시판 URL에서 게시글 제목/링크 목록을 CSV로 저장한다.

```bash
# 기본값: 게시판 40번 메뉴 → data/raw/csv/cafe_articles.csv
bash script/run_crawler.sh

# 출력 CSV와 게시판 URL을 직접 지정
bash script/run_crawler.sh data/raw/csv/cafe_menus_40.csv \
  "https://cafe.naver.com/f-e/cafes/25741302/menus/41"
```

**2단계: 게시글 상세 수집** - 두 가지 방식 중 선택

개별 게시글 1건 수집 (`script/run_article.sh`):

```bash
bash script/run_article.sh \
  "https://cafe.naver.com/f-e/cafes/25741302/articles/44443?menuid=112"
```

CSV 기반 일괄 수집 (`script/run_articles_from_csv.py`):

```bash
# 1단계에서 만든 CSV의 모든 게시글을 순회하며 JSON으로 저장
uv run python script/run_articles_from_csv.py \
  data/raw/csv/cafe_menus_112.csv \
  data/raw/articles
```

### 파이프라인 실행

```bash
# Stage 1: 규칙 기반 시나리오 추출
python -m app.customer_research.pipeline.extract_scenarios

# Stage 2: LLM 질문 요약
python -m app.customer_research.pipeline.summarize_with_llm

# Stage 3: LLM 구조화 추출
python -m app.customer_research.pipeline.extract_to_state

# Stage 4: 통계 기반 필드 보정
python -m app.customer_research.pipeline.fill_required_fields
```

`extract_to_state` 주요 옵션:

| 옵션 | 설명 |
|------|------|
| `--sample N` | 랜덤 N개 게시글만 처리 |
| `--ids ID1 ID2` | 특정 게시글만 처리 |
| `--output-dir PATH` | 출력 디렉토리 지정 |
| `--resume` | 기존 출력 파일 기준으로 건너뛰고 이어서 처리 |

---

## 참조목록

| 파일 | 설명 |
|------|------|
| `app/customer_research/feature/http_client.py` | HTTP 요청 유틸리티 (`fetch_html`, `fetch_html_safe`) |
| `app/customer_research/feature/naver_cafe_parser.py` | 카페 목록 HTML 파싱 (`get_article_titles_from_html`, `get_next_page_url`) |
| `app/customer_research/feature/naver_article_parser.py` | 게시글 상세 파싱 (`parse_article_html`, `fetch_and_parse_article`) |
| `app/customer_research/feature/naver_crawler.py` | 게시판 크롤링 (`get_cafe_menu_article_titles`) |
| `app/customer_research/pipeline/__init__.py` | 파이프라인 스테이지 정의 |
| `app/customer_research/pipeline/extract_scenarios.py` | 규칙 기반 시나리오 추출 (`create_scenario_from_article`, `main`) |
| `app/customer_research/pipeline/summarize_with_llm.py` | LLM 요약 (`summarize_with_llm`, `process_scenarios`) |
| `app/customer_research/pipeline/extract_to_state.py` | LLM 구조화 추출 (`call_llm`, `validate_and_fix`, `split_and_save`) |
| `app/customer_research/pipeline/fill_required_fields.py` | 통계 기반 필드 보정 (`process_scenario`, `main`) |
| `script/run_crawler.sh` | 게시판 목록 크롤링 실행 스크립트 |
| `script/run_article.sh` | 개별 게시글 상세 수집 실행 스크립트 |
| `script/run_articles_from_csv.py` | CSV 기반 게시글 일괄 수집 스크립트 |
