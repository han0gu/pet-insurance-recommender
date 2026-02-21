from langchain_core.documents import Document

chunk = Document(
    page_content=('- 조 (특별약관의 재가입에 관한 사항) 제5항에 따라 보험계약이 연장된 경우에는 적용\n'
 '- 하지 않습니다.\n'
 '- ⑤ 회사가 지급할 제1항에서 정한 의료비보험금(수술당일)은 보험증권에 기재된 자기부\n'
 '- 담금을 차감한 후 보상비율을 곱한 금액이며 보험증권에 기재된 1일당 보상한도액을\n'
 '- 한도로 합니다.(자기부담급은 수술 당일 의료비에서 차감합니다.)\n'
 '<지급보험금의 계산>{(피보험자가 부담한 수술 당일 의료비 – 1일당 자기부담금) × 보상비율}과 보험증권에 기재된 1'),
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
