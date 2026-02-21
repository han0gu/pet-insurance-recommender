from langchain_core.documents import Document

chunk = Document(
    page_content=('범위 내일 것\n'
 '2. 재가입 전 계약의 보험료가 정상적으로 납입완료 되었을 것- \n'
 '② 이 계약의 보험기간 종료 후 계약자가 재가입을 원하는 경우 계약자는 재가입 시점에\n'
 '서 회사가 판매하는 동일하거나 객관적이고 합리적인 범위내에서 기존 계약내용에 상\n'
 '응한 반려동물보험 상품(보험업감독규정 제1-2조(정의)에서 정한 장기손해보험에 한\n'
 '하며 이하「반려동물보험 상품」이라 합니다)으로 가입을 할 수 있으며, 회사는 이를\n'
 '거절할 수 없습니다. 다만, 재가입 계약이 직전계약보다 보장내용 및 범위 등이 확대'),
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
