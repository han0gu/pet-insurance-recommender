#!/usr/bin/env bash
# 사용법: run_crawler.sh [출력CSV] [게시판URL]
# 예: run_crawler.sh
#     run_crawler.sh data/raw/csv/cafe_menus_40.csv
#     run_crawler.sh out.csv "https://cafe.naver.com/f-e/cafes/25741302/menus/41"
set -e
cd "$(dirname "$0")/.."
OUTPUT_CSV="${1:-data/raw/csv/cafe_articles.csv}"
CAFE_URL="${2:-https://cafe.naver.com/f-e/cafes/25741302/menus/40}"
uv run python -c "
import csv
from app.customer_research.feature.naver_crawler import get_cafe_menu_article_titles

url = \"$CAFE_URL\"
items = get_cafe_menu_article_titles(url, use_browser=True)
with open(\"$OUTPUT_CSV\", \"w\", newline=\"\", encoding=\"utf-8\") as f:
    w = csv.writer(f)
    w.writerow([\"index\", \"글번호\", \"제목\", \"링크\"])
    for i, (article_id, title, link) in enumerate(items, 1):
        w.writerow([i, article_id or \"\", title, link])
print(\"게시글\", len(items), \"건 저장:\", \"$OUTPUT_CSV\")
"
