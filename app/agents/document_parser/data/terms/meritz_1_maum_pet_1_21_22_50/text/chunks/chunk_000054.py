from langchain_core.documents import Document

chunk = Document(
    page_content=('사실을 안 날부터 1개월 이내에 이 계약을 해지할 수 있습니다.1. 계약자, 피보험자 또는 이들의 대리인이 고의 또는 중대한 과실로 '
 '제15조(계약 전\n'
 '알릴 의무)를 위반하고 그 의무가 중요한 사항에 해당하는 경우\n'
 '2. 뚜렷한 위험의 증가와 관련된 제16조(계약 후 알릴 의무) 제1항에서 정한 계약 후\n'
 '알릴 의무를 계약자, 피보험자 또는 이들의 대리인의 고의 또는 중대한 과실로 이행\n'
 '하지 않았을때- 10 -② 제1항 제1호의 경우에도 불구하고 다음 중 하나에 해당하는 경우에는 회사는 계약을'),
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
