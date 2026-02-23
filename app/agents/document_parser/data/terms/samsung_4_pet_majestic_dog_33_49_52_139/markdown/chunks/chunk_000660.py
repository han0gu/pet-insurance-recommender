from langchain_core.documents import Document

chunk = Document(
    page_content=('- 위권을 포기합니다.\n'
 '- ④ 회사는 제1항에 따른 권리가 계약자 또는 피보험자와 생계를 같이 하는 가족에 대한\n'
 '- 것인 경우에는 그 권리를 취득하지 못합니다. 다만, 손해가 그 가족의 고의로 인하여\n'
 '- 발생한 경우에는 그 권리를 취득합니다.\n'
 '# 제 12조 (특별약관의 소멸)- ① 보험증권에 기재된 반려견이 보험기간 중에 사망하여 보험의 목적에 대해 이 특별약\n'
 '- 관에서 정한 보험금 지급사유가 더이상 발생할 수 없는 경우에는 "보험료 및 해약환\n'
 '- 급금 산출방법서" 에 정하는 바에 따라 회사가 적립한 사망당시 이 특별약관의 계약'),
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
