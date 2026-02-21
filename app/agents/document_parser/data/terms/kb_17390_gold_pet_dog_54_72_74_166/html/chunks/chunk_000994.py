from langchain_core.documents import Document

chunk = Document(
    page_content=('rowspan="2">특정처치(이물제거)</td><td>이물제거(내시경)</td><td '
 'rowspan="2">연간2회한</td><td>200만원</td></tr><tr><td>이물제거(구토유도약물)</td><td>20만원</td></tr><tr><td '
 'colspan="2">창상/교상치료</td><td>연간1회한</td><td>70만원</td></tr><tr><td '
 'colspan="2">특정약물치료Ⅱ</td><td>연간12회한</td><td>10만원</td></tr><tr><td'),
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
