from langchain_core.documents import Document

chunk = Document(
    page_content=('| --- | --- | --- |\n'
 '| \uf000 약관에 규정하는 | 6대호흡계특정질환으로 질병은 | 분류되는 제9차 개정 한국표준질 |\n'
 '병 ․사인분류(KCD, 통계청 고시 제2025-299호, 2026.1.1. 시행) 중 다음에 적은 질 규정\n'
 '병을 말하며 이후 한국표준질병․사인분류가 개정되는 경우는 개정된 기준에 따라\n'
 '이 약관에서 보장하는 6대호흡계특정질환 해당 여부를 판단합니다.\n'
 '대상이 되는 항목 분류번호ㆍ| 아데노바이러스폐렴 특정 | 아데노바이러스폐렴 특정 | J12.0 |\n'
 '| --- | --- | --- |'),
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
