from langchain_core.documents import Document

chunk = Document(
    page_content=('익자의 책임있는 사유로 지급이 지연된 때에는 그 해당기간에 대한 이자는 더하여 지\n'
 '급하지 않습니다.- ⑤ 계약자, 피보험자 또는 보험수익자는 제13조(알릴 의무 위반의 효과) 및 제2항의 보험\n'
 '- 금 지급사유조사와 관련하여 의료기관, 동물병원, 국민건강보험공단, 경찰서 등 관공\n'
 '- 서에 대한 회사의 서면에 의한 조사요청에 동의하여야 합니다. 다만, 정당한 사유없이\n'
 '- 이에 동의하지 않을 경우 사실확인이 끝날 때까지 회사는 보험금 지급 지연에 따른\n'
 '- 이자를 지급하지 않습니다.'),
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
