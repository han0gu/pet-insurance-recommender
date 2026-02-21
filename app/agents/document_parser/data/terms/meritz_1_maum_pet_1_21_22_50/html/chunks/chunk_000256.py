from langchain_core.documents import Document

chunk = Document(
    page_content=('. 이에 필요한 비용은 회사가 지급합니다.<br>③ 회사는 제1항 및 제2항에도 불구하고 타인을 위한 보험계약의 경우에는 계약자에 '
 '대한<br>대위권을 포기합니다.<br>④ 회사는 제1항에 따른 권리가 계약자 또는 피보험자와 생계를 같이 하는 가족에 대한<br>것인 '
 '경우에는 그 권리를 취득하지 못합니다'),
    metadata={'source_doc': {'total_pages': 50},
 'doc': {'doc_type': 'terms',
         'file_name': 'meritz_1_maum_pet_1_21_22_50.pdf',
         'insurer_code': 'meritz',
         'product_code': '1',
         'product_name': '메리츠 마음든든 반려동물보험',
         'total_pages': 1,
         'page': 1},
 'term_type': 'unknown'},
)
