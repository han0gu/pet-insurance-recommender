from __future__ import annotations

from typing import List

from langchain_core.documents import Document


product_name_by_code = {
    ("meritz", "1"): "메리츠 마음든든 반려동물보험",
    ("meritz", "2"): "무배당 펫퍼민트 Cat&Family보험 다이렉트2601",
    ("meritz", "3"): "무배당 펫퍼민트 Puppy&Family보험 다이렉트2601",
    ("samsung", "1"): "(일반)반려견보험 애니펫",
    ("samsung", "2"): "(일반)반려묘보험 애니펫",
    ("samsung", "3"): "(장기)무배당 삼성화재 다이렉트 착한펫보험(강아지)",
    ("samsung", "4"): "(장기)무배당 삼성화재 펫보험 위풍댕댕",
    ("samsung", "5"): "(장기)무배당 삼성화재 펫보험 의기냥냥",
    ("kb", "17390"): "[일반보험] KB반려행복펫보험",
    ("kb", "25048"): "[상해보험] KB 금쪽같은 펫보험(강아지)(무배당)(26.01)",
    ("kb", "25049"): "[제휴] KB 금쪽같은 펫보험(강아지)(무배당)(26.01)",
    ("kb", "25050"): "[상해보험] KB 금쪽같은 펫보험(고양이)(무배당)(26.01)",
    ("kb", "25051"): "[제휴] KB 금쪽같은 펫보험(고양이)(무배당)(26.01)",
    ("kb", "25064"): "[제휴] KB 다이렉트 금쪽같은 펫보험(강아지)(무배당)(26.01)",
    ("kb", "25065"): "[제휴] KB 다이렉트 금쪽같은 펫보험(고양이)(무배당)(26.01)",
}


def add_metadata_to_documents(
    *,
    documents: List[Document],
    file_name: str,
    basic_term_start: int,
    basic_term_end: int,
    special_term_start: int,
    special_term_end: int,
    page_numbers: List[int] | None = None,
    source_document_metadata: dict | None = None,
) -> List[Document]:
    if not documents:
        return []

    if page_numbers is not None and len(page_numbers) != len(documents):
        raise ValueError("page_numbers length must match documents length")

    source_document_metadata = source_document_metadata or {}
    total_pages = len(documents)

    insurer_code = file_name.split("_")[0]
    product_code = file_name.split("_")[1]
    product_name = product_name_by_code[(insurer_code, product_code)]

    resolved_pages = page_numbers or list(range(1, total_pages + 1))
    split_ok = max(resolved_pages) > 1

    result: List[Document] = []
    for idx, document in enumerate(documents):
        page_number = resolved_pages[idx]
        term_type = _resolve_term_type(
            page_number=page_number,
            split_ok=split_ok,
            basic_term_start=basic_term_start,
            basic_term_end=basic_term_end,
            special_term_start=special_term_start,
            special_term_end=special_term_end,
        )

        new_metadata = {
            "source_doc": dict(source_document_metadata) or dict(document.metadata or {}),
            "doc": {
                "doc_type": "terms",
                "file_name": file_name,
                "insurer_code": insurer_code,
                "product_code": product_code,
                "product_name": product_name,
                "total_pages": total_pages,
                "page": page_number,
            },
            "term_type": term_type,
        }
        result.append(Document(page_content=document.page_content, metadata=new_metadata))

    return result


def _resolve_term_type(
    *,
    page_number: int,
    split_ok: bool,
    basic_term_start: int,
    basic_term_end: int,
    special_term_start: int,
    special_term_end: int,
) -> str:
    if not split_ok:
        return "unknown"
    if basic_term_start <= page_number <= basic_term_end:
        return "basic"
    if special_term_start <= page_number <= special_term_end:
        return "special"
    return "unknown"
