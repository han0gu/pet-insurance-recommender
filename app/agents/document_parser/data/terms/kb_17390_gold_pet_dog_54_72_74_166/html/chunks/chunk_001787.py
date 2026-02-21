from langchain_core.documents import Document

chunk = Document(
    page_content=('판단된 경우, 이후 한국표준질병․사인분류 개정으로 질병 분류가<br>변경되더라도 이 약관에서 보장하는 질병 해당 여부를 다시 판단하지 '
 "않습니다.</p><p id='107' data-category='paragraph' style='font-size:16px'>별표15 "
 '환경성질환 분류표<br>\uf000 약관에 규정하는 환경성질환으로 분류되는 질병은 제9차 개정 한국표준질병․사인<br>분류(KCD, '
 '통계청 고시 제2025-299호, 2026.1.1'),
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
