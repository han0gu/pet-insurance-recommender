from __future__ import annotations

import re
from dataclasses import dataclass
from typing import List

from bs4 import BeautifulSoup, Tag

from langchain_core.documents import Document


product_name_by_code = {
    ("meritz", "1"): "메리츠 마음든든 반려동물보험",
    ("meritz", "2"): "무배당 펫퍼민트 Puppy&Family보험 다이렉트2601",
    ("meritz", "3"): "무배당 펫퍼민트 Cat&Family보험 다이렉트2601",
}


@dataclass
class PageDoc:
    page: int
    text: str
    anchor_ids: List[str]


_PAGE_RE = re.compile(r"^\s*(?:-\s*)?(\d{1,4})(?:\s*-\s*)?\s*$")


def _norm_text(s: str) -> str:
    s = s.replace("\u00a0", " ")
    s = re.sub(r"[ \t]+", " ", s)
    s = re.sub(r"\n{3,}", "\n\n", s)
    return s.strip()


def _extract_table_text(table_tag: Tag) -> str:
    """
    <table> → 행(row) 단위 텍스트로 변환
    각 셀(cell)은 ' | ' 로 연결
    """
    rows = []

    for tr in table_tag.find_all("tr", recursive=True):
        cells = []
        for cell in tr.find_all(["td", "th"], recursive=False):
            text = cell.get_text(" ", strip=True)
            if text:
                cells.append(text)

        if cells:
            row_text = " | ".join(cells)
            rows.append(row_text)

    return "\n".join(rows)


def _dedup_keep_order(items: List[str]) -> List[str]:
    seen = set()
    unique: List[str] = []
    for item in items:
        if item in seen:
            continue
        seen.add(item)
        unique.append(item)
    return unique


def split_pages_and_add_metadata(
    full_document: Document,
    file_name: str,
) -> List[Document]:
    """
    Upstage Document Parse 결과(단일 HTML Document)를 페이지 footer 기준으로 분리한다.
    """
    html = full_document.page_content
    soup = BeautifulSoup(html, "html.parser")

    root = soup.body if soup.body else soup

    page_docs: List[PageDoc] = []
    buf_text: List[str] = []
    buf_ids: List[str] = []

    for node in root.descendants:
        if not isinstance(node, Tag):
            continue

        # -------------------------
        # 1. 페이지 마커 처리
        # -------------------------
        if node.name == "footer":
            raw = node.get_text(strip=True)
            m = _PAGE_RE.match(raw)
            if m:
                page_num = int(m.group(1))
                text = _norm_text("\n".join(buf_text))

                page_docs.append(
                    PageDoc(
                        page=page_num,
                        text=text,
                        anchor_ids=_dedup_keep_order(buf_ids),
                    )
                )

                buf_text.clear()
                buf_ids.clear()

            continue

        # -------------------------
        # 2. Table 처리 (중복 방지)
        # -------------------------
        if node.name == "table":
            table_text = _extract_table_text(node)
            if table_text:
                buf_text.append(table_text)

            node_id = node.get("id")
            if node_id:
                buf_ids.append(str(node_id))

            # table 내부는 여기서 이미 처리했으므로
            # descendants 순회 중복 방지 위해 skip
            continue

        # -------------------------
        # 3. 일반 텍스트 태그 처리
        # -------------------------
        if node.name in {
            "p",
            "h1",
            "h2",
            "h3",
            "li",
            "div",
            "span",
        }:
            text = node.get_text(" ", strip=True)
            if text:
                buf_text.append(text)

            node_id = node.get("id")
            if node_id:
                buf_ids.append(str(node_id))

    # 마지막 페이지 footer가 누락된 경우를 대비한 fallback
    if buf_text:
        inferred_page = page_docs[-1].page + 1 if page_docs else 1
        page_docs.append(
            PageDoc(
                page=inferred_page,
                text=_norm_text("\n".join(buf_text)),
                anchor_ids=_dedup_keep_order(buf_ids),
            )
        )

    total_pages = len(page_docs)
    insurer_code = file_name.split("_")[0]
    product_code = file_name.split("_")[1]
    product_name = product_name_by_code[
        (file_name.split("_")[0], file_name.split("_")[1])
    ]
    return [
        Document(
            page_content=page_doc.text,
            metadata={
                **full_document.metadata,
                "doc": {
                    "doc_type": "terms",
                    "file_name": file_name,
                    "insurer_code": insurer_code,
                    "product_code": product_code,
                    "product_name": product_name,
                    "total_pages": total_pages,
                    "page": page_doc.page,
                    "anchor_ids": page_doc.anchor_ids,
                },
            },
        )
        for page_doc in page_docs
    ]
