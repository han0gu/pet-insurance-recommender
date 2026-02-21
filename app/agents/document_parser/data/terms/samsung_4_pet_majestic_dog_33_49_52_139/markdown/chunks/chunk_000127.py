from langchain_core.documents import Document

chunk = Document(
    page_content=('- 를 한 피보험자는 계약의 효력이 유지되는 기간에는 언제든지 서면동의를 장래를 향\n'
 '- 하여 철회할 수 있으며, 서면동의 철회로 계약이 해지되어 회사가 지급하여야 할 해약\n'
 '- 환급금이 있을 때에는 제36조(해약환급금) 제1항에 따른 해약환급금을 계약자에게 지\n'
 '- 급합니다.\n'
 '# 제 33조의2 (위법계약의 해지)① 계약자는 「금융소비자 보호에 관한 법률」제47조 및 관련규정이 정하는 바에 따라\n'
 '계약체결에 대한 회사의 법위반사항이 있는 경우 계약체결일부터 5년 이내의 범위에'),
    metadata={'source_doc': {'total_pages': 107},
 'doc': {'doc_type': 'terms',
         'file_name': 'samsung_4_pet_majestic_dog_33_49_52_139.pdf',
         'insurer_code': 'samsung',
         'product_code': '4',
         'product_name': '(장기)무배당 삼성화재 펫보험 위풍댕댕',
         'total_pages': 1,
         'page': 1},
 'term_type': 'unknown'},
)
