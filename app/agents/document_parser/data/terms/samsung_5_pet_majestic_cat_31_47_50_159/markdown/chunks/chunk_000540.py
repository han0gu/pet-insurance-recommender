from langchain_core.documents import Document

chunk = Document(
    page_content=('- 계약자의 재가입의사가 확인되지 않는 경우 계약이 해지된다는 사실을 알려드립니다.\n'
 '- ⑧ 제7항에 따라 계약자에게 해지된다는 사실을 알려드린 최초시점부터 90일 이내에 계\n'
 '- 약자의 재가입 의사가 확인되지 않는 경우 해당 시점부터 계약은 해지됩니다.\n'
 '- ⑨ 제5항에 따라 보험계약이 연장된 경우 계약자는 회사에 재가입 의사를 표시할 수 있\n'
 '- 습니다. 회사는 계약자의 재가입 의사가 확인되었을 때에는 제1항 및 제2항에서 정한\n'
 '- 절차에 따라 회사가 재가입 의사를 확인한 날에 판매중인 제2항의 반려동물보험 상품'),
    metadata={'source_doc': {'total_pages': 129},
 'doc': {'doc_type': 'terms',
         'file_name': 'samsung_5_pet_majestic_cat_31_47_50_159.pdf',
         'insurer_code': 'samsung',
         'product_code': '5',
         'product_name': '(장기)무배당 삼성화재 펫보험 의기냥냥',
         'total_pages': 1,
         'page': 1},
 'term_type': 'unknown'},
)
