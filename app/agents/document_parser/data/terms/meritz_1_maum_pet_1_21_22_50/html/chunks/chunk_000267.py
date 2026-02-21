from langchain_core.documents import Document

chunk = Document(
    page_content=('. 계약자, 피보험자 또는 이들의 대리인이 고의 또는 중대한 과실로 보통약관 제15조<br>(계약 전 알릴 의무)를 위반하고 그 의무가 '
 '중요한 사항에 해당하는 경우<br>2. 뚜렷한 위험의 증가와 관련된 제15조(계약 후 알릴 의무) 제1항에서 정한 계약 후<br>알릴 '
 '의무를 계약자 또는 피보험자의 고의 또는 중대한 과실로 이행하지 않았을 때<br>3'),
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
