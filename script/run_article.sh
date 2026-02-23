#!/usr/bin/env bash
# 게시글 상세 파싱: 제목, 본문, 첨부 이미지, 댓글 → 글번호.json
# 사용법: run_article.sh <게시글URL> [출력디렉토리]
# 예: run_article.sh "https://cafe.naver.com/f-e/cafes/25741302/articles/44443?menuid=112"
#     run_article.sh "https://..." data/raw/articles
set -e
cd "$(dirname "$0")/.."
ARTICLE_URL="${1:?게시글 URL을 입력하세요}"
OUTPUT_DIR="${2:-data/raw/articles}"
uv run python -c "
from app.customer_research.feature.naver_article_parser import fetch_and_parse_article, save_article_json

data = fetch_and_parse_article(\"$ARTICLE_URL\", use_browser=True)
path = save_article_json(data, \"$OUTPUT_DIR\")
print(\"저장:\", path)
print(\"제목:\", (data.get(\"title\") or \"\")[:60])
print(\"댓글 수:\", len(data.get(\"comments\") or []))
print(\"첨부 수:\", len(data.get(\"attachments\") or []))
"
