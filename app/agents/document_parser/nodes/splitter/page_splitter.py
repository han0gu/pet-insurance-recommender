from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import List

from bs4 import BeautifulSoup, Tag

from langchain_core.documents import Document
from langchain_upstage.document_parse import OutputFormat

from app.agents.document_parser.constants import TERMS_DIR


product_name_by_code = {
    ("meritz", "1"): "메리츠 마음든든 반려동물보험",
    ("meritz", "2"): "무배당 펫퍼민트 Puppy&Family보험 다이렉트2601",
    ("meritz", "3"): "무배당 펫퍼민트 Cat&Family보험 다이렉트2601",
}

output_extension_by_format = {
    "html": "html",
    "text": "txt",
    "markdown": "md",
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
    *,
    basic_term_start: int,
    basic_term_end: int,
    special_term_start: int,
    special_term_end: int,
    output_format: OutputFormat = "html",
) -> List[Document]:
    """
    Upstage Document Parse 결과(단일 HTML Document)를
    footer의 페이지 마커 기준으로 분리하고
    각 페이지에 문서/약관 메타데이터를 추가한다.

    Args:
        full_document: Upstage Document Parser가 반환한 단일 HTML Document.
        file_name: 원본 PDF 파일명(확장자 포함).
        basic_term_start: 보통약관 시작 페이지 번호(문서 footer 기준).
        basic_term_end: 보통약관 종료 페이지 번호(문서 footer 기준).
        special_term_start: 특별약관 시작 페이지 번호(문서 footer 기준).
        special_term_end: 특별약관 종료 페이지 번호(문서 footer 기준).
        output_format: 페이지별 로컬 저장 파일 포맷(`html`, `text`, `markdown`).

    Returns:
        페이지별로 분리되고 metadata가 추가된 `Document` 리스트.

    Raises:
        ValueError: 필수 인자가 비어 있거나 유효하지 않은 경우.
    """

    if (
        not full_document
        or not file_name
        or not basic_term_start
        or not basic_term_end
        or not special_term_start
        or not special_term_end
    ):
        raise ValueError("❗️split_pages_and_add_metadata invalid params")

    result: List[Document] = []

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

    for page_doc in page_docs:
        if basic_term_start <= page_doc.page <= basic_term_end:
            term_type = "basic"
        elif special_term_start <= page_doc.page <= special_term_end:
            term_type = "special"
        else:
            term_type = ""

        new_doc = Document(
            page_content=page_doc.text,
            metadata={
                "source": {**full_document.metadata},
                "doc": {
                    "doc_type": "terms",  # 약관
                    "file_name": file_name,  # 약관 파일명 (확장자 포함)
                    "insurer_code": insurer_code,  # 보험사 코드 (e.g. samsung, kb, meritz, ...)
                    "product_code": product_code,  # 보험 상품 코드 (e.g. 1, 2, 3, ...)
                    "product_name": product_name,  # 보험 상품 명 (e.g. 메리츠 마음든든 반려동물보험)
                    "total_pages": total_pages,  # 총 페이지 수
                    "page": page_doc.page,  # 현재 페이지 번호
                    "anchor_ids": page_doc.anchor_ids,  # 파싱 과정에서 식별되는 HTML 태그 ID (e.g. id="23")
                },
                "term_type": term_type,  # 약관 유형 (e.g. basic: 보통약관, special: 특별약관)'
            },
        )
        result.append(new_doc)

        DP_RESULTS_DIR = TERMS_DIR / file_name.split(".")[0]
        create_page_content_file(new_doc, DP_RESULTS_DIR, output_format)

    return result


def create_page_content_file(
    doc: Document,
    target_dir: Path,
    output_format: OutputFormat = "html",
):
    # rprint("create_page_content_file target_dir:", target_dir)
    target_dir.mkdir(parents=True, exist_ok=True)

    file_name_without_extension = doc.metadata["doc"]["file_name"].split(".")[0]
    OUTPUT_EXTENSION = output_extension_by_format.get(output_format)
    OUTPUT_FILE_NAME = f"{file_name_without_extension}_{doc.metadata['doc']['page']}.{OUTPUT_EXTENSION}"
    OUTPUT_FILE_PATH = target_dir / OUTPUT_FILE_NAME
    # rprint("🔗create_local_file OUTPUT_FILE_PATH:", OUTPUT_FILE_PATH)
    if OUTPUT_FILE_PATH.exists():
        # rprint("⚠️ create_local_file skipped (already exists)")
        return

    # rprint("🚀create_local_file start")
    OUTPUT_FILE_PATH.write_text(doc.page_content, encoding="utf-8")
    # rprint("✅create_local_file done")
