from langchain_core.documents import Document

chunk = Document(
    page_content=('의무 등)를 준용하여 회사가 정한 절차에 따라 계약자는 기존 계약에 이어 재가입할\n'
 '수 있으며, 이 경우 회사는 기존계약의 가입 이후 발생한 반려동물의 상해 또는 질병\n'
 '을 사유로 가입을 거절할 수 없습니다. 단, 특별약관 일반사항의 제19조(특별약관의\n'
 '성립) 제1항 및 제2항에도 불구하고 제2항에서 말하는 별도의 반려동물보험 상품으로\n'
 '체결될 수 있습니다.1. 재가입일에 있어서 반려동물의 나이가 회사가 최초가입 당시 정한 재가입 나이의\n'
 '범위 내일 것\n'
 '2. 재가입 전 계약의 보험료가 정상적으로 납입완료 되었을 것-'),
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
