#!/usr/bin/env python3
"""CSV에 나열된 모든 게시글 URL에 대해 상세 파싱을 수행하고 JSON으로 저장한다.

사용법:
  uv run python scripts/run_articles_from_csv.py [CSV경로] [출력디렉토리]
  uv run python scripts/run_articles_from_csv.py data/raw/csv/cafe_menus_112.csv data/raw/articles

기본: CSV=data/raw/csv/cafe_menus_112.csv, 출력=data/raw/articles
"""

import csv
import sys
from pathlib import Path

from app.customer_research.feature.naver_article_parser import fetch_and_parse_article, save_article_json


def main() -> None:
    project_root = Path(__file__).resolve().parent.parent
    csv_path = Path(sys.argv[1]) if len(sys.argv) > 1 else project_root / "data" / "raw" / "csv" / "cafe_menus_112.csv"
    output_dir = Path(sys.argv[2]) if len(sys.argv) > 2 else project_root / "data" / "raw" / "articles"

    if not csv_path.is_file():
        print(f"오류: CSV 파일을 찾을 수 없습니다. {csv_path}", file=sys.stderr)
        sys.exit(1)

    output_dir.mkdir(parents=True, exist_ok=True)
    rows = list(csv.DictReader(csv_path.open(encoding="utf-8")))
    link_col = "링크"
    if not rows or link_col not in rows[0]:
        print(f"오류: CSV에 '{link_col}' 열이 없습니다.", file=sys.stderr)
        sys.exit(1)

    total = len(rows)
    ok = 0
    fail = 0
    for i, row in enumerate(rows, start=1):
        url = (row.get(link_col) or "").strip()
        if not url or not url.startswith("http"):
            fail += 1
            print(f"[{i}/{total}] URL 없음/건너뜀")
            continue
        try:
            data = fetch_and_parse_article(url, use_browser=True)
            path = save_article_json(data, output_dir)
            aid = data.get("article_id", "?")
            ok += 1
            print(f"[{i}/{total}] 저장: {path} (article_id={aid})")
        except Exception as e:
            fail += 1
            print(f"[{i}/{total}] 실패: {url[:60]}... — {e}", file=sys.stderr)

    print(f"완료: 성공 {ok}, 실패 {fail}, 총 {total}")


if __name__ == "__main__":
    main()
