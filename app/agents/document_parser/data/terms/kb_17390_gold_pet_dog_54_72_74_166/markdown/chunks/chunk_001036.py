from langchain_core.documents import Document

chunk = Document(
    page_content=('| 심낭수종 | 803 순환기계질환 |  |\n'
 '| 심내막염 | 803 순환기계질환 |  |\n'
 '| 심방중격결손 | 803 순환기계질환 |  |\n'
 '| 심부전 심비대 | 803 순환기계질환 |  |\n'
 '| 심실중격결손 | 803 순환기계질환 |  |\n'
 '| 심정지 | 803 순환기계질환 |  |\n'
 '| 우대동맥궁 잔존 | 803 순환기계질환 |  |\n'
 '| 이첨판폐쇄부전 | 803 순환기계질환 |  |\n'
 '| 폐동맥협착증 | 803 순환기계질환 |  |\n'
 '| 폐성고혈압 혈전 / 색전증 | 803 순환기계질환 |  |'),
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
