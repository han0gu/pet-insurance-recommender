from langchain_core.documents import Document

chunk = Document(
    page_content=('- 및 중도인출금에 부리되었을 이자만큼 해약환급금에서 차감하여 계산하므로 제1항에\n'
 '- 정한 지급금이 감소합니다.\n'
 '- ④ 제24조(계약내용의 변경 등) 제1항 제5호에서 정한 적립보험료 등을 감액할 경우 제1\n'
 '- 항에 정한 해약환급금은 없거나 최초가입시 안내한 금액보다 적어질 수 있습니다.\n'
 '- ⑤ 회사는 경과기간별 해약환급금에 관한 표를 계약자에게 제공하여 드립니다.\n'
 '- ⑥ 제33조의2(위법계약의 해지)에 따라 위법계약이 해지되는 경우 회사가 적립한 해지\n'
 '- 당시의 계약자적립액 및 미경과보험료를 반환하여 드립니다.'),
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
