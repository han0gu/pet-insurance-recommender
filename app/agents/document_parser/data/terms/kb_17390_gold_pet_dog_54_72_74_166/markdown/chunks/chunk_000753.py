from langchain_core.documents import Document

chunk = Document(
    page_content='| 퍼스널모빌리티(세그웨이, | 전동킥보드, 전동이륜평행차 등)는 | 자동차관리법에 |\n| --- | --- | --- |',
    metadata={'source_doc': {'total_pages': 113},
 'doc': {'doc_type': 'terms',
         'file_name': 'kb_17390_gold_pet_dog_54_72_74_166.pdf',
         'insurer_code': 'kb',
         'product_code': '17390',
         'product_name': '[일반보험] KB반려행복펫보험',
         'total_pages': 1,
         'page': 1},
 'term_type': 'unknown'},
)
