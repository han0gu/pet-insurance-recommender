from langchain_core.documents import Document

chunk = Document(
    page_content=('- 가입에 관한 사항) 제1항 및제2항에 따라 재가입하는 경우또는 제27조 (특별약관의\n'
 '- 재가입에 관한 사항) 제5항에 따라 보험계약이 연장된 경우에는 적용하지 않습니다.\n'
 '- ⑤ 회사가 지급할 제1항에서 정한 의료비보험금(수술당일제외, 검사비포함)은 보험증권에\n'
 '- 기재된 자기부담금을 차감한 후 보상비율을 곱한 금액이며 보험증권에 기재된 1일당\n'
 '- 보상한도액을 한도로 합니다. (자기부담금은 1일당 의료비에서 차감합니다.)'),
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
