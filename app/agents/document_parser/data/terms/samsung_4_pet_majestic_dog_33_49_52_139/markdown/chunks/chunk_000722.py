from langchain_core.documents import Document

chunk = Document(
    page_content=('- ③ 제1항 제2호에도 불구하고 계약 전 알릴 의무를 위반하고 계약자가 보험계약의 변경\n'
 '- 에 대한 청약을 하지 않는 경우 회사는 보통약관 「계약 전 알릴 의무 위반의 효과」\n'
 '- 조항에 따라 보험계약을 해지할 수 있습니다.\n'
 '- ④ 이 특별약관에 대한 회사의 보장개시일(책임개시일)은 보험계약 「제1회 보험료 및 회\n'
 '- 사의 보장개시」에서 정한 보장개시일(책임개시일)과 동일합니다.\n'
 '- ⑤ 보험계약이 해지, 기타 사유에 의하여 효력이 없게 된 경우에는 이 특별약관도 더 이\n'
 '- 상 효력이 없습니다.'),
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
