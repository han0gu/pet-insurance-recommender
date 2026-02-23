from langchain_core.documents import Document

chunk = Document(
    page_content=('| 폐성고혈압 혈전 / 색전증 | 803 순환기계질환 |  |\n'
 '| 기타 선천성 순환기계 질환 | 803 순환기계질환 |  |\n'
 '| 기타 | 803 순환기계질환 |  |\n'
 '| 심질환 기타 혈관 질환 | 803 순환기계질환 |  |\n'
 '| 기타 림프계 질환 | 803 순환기계질환 |  |\n'
 '| 기타 순환기계 질환 | 803 순환기계질환 |  |\n'
 '|  | 803 순환기계질환 |  |\n'
 '| 부정맥 | 803 순환기계질환 |  |\n'
 '| 코드 | 특정 질병 | 세부 질병명 |\n'
 '| --- | --- | --- |\n'
 '| 804 | 안과질환 |  |'),
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
