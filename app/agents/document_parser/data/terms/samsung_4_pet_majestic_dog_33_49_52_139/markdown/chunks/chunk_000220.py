from langchain_core.documents import Document

chunk = Document(
    page_content=('으로 봅니다.<용어풀이>[약관의 중요한 내용]\n'
 '금융소비자 보호에 관한 법률 제19조(설명의무) 등에서 정한 다음의 내용을 말합니다.- - 보험금 지급제한 사유 및 지급절차\n'
 '- - 청약의 철회에 관한 사항\n'
 '- - 계약의 해지 및 해제\n'
 '- - 분쟁조정 절차에 관한 사항\n'
 '- - 예금자보호법에 따른 보호여부\n'
 '- - 환급금에 관한 사항\n'
 '- - 고지의무 및 통지의무 위반의 효과\n'
 '- - 저축성 보험계약의 경우 적용이율 및 산출기준\n'
 '- - 유배당 보험계약의 경우 계약자 배당에 관한 사항\n'
 '- - 만기시 자동갱신되는 보험계약의 경우 자동갱신의 조건'),
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
