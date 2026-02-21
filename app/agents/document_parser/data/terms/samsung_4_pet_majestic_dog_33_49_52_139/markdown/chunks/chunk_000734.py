from langchain_core.documents import Document

chunk = Document(
    page_content=('- 계속하여 입원한 경우 그 입원에 대해서는 회사가 보험금을 지급하지 않는 기간 종료\n'
 '- 일의 다음날을 입원의 개시일로 인정하여 보험금을 지급합니다.\n'
 '- ⑧ 피보험자에게 보험금의 지급사유 또는 보험료 납입면제사유가 발생했을 경우, 그 보험\n'
 '- 금의 지급사유 또는 보험료 납입면제사유가 특정신체부위 또는 특정질병을 직접적인\n'
 '- 원인으로 발생한 사고인가 아닌가는 의사의 진단서와 의견을 주된 판단자료로 결정합\n'
 '- 니다.\n'
 '- ⑨ 제1항의 특정신체부위와 특정질병은 4개 이내에서 선택하여 부가할 수 있습니다.'),
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
