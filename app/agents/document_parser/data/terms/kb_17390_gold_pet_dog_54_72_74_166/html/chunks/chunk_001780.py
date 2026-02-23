from langchain_core.documents import Document

chunk = Document(
    page_content=("않습니다.</h1><p id='101' data-category='paragraph' style='font-size:14px'>별표14 "
 '호흡기관련질병 분류표<br>\uf000 약관에 규정하는 호흡기관련질병은 제9차 개정 한국표준질병․사인분류(KCD, 통계<br>청 고시 '
 '제2025-299호, 2026.1.1'),
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
