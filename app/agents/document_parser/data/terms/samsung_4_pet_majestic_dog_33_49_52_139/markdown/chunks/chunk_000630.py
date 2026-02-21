from langchain_core.documents import Document

chunk = Document(
    page_content=('- ③ 보험의 목적이 다수인 경우 제1항 내지 제2항은 보험의 목적별로 각각 적용합니다.\n'
 '제6조 (보험료의 납입을 연체하여 해지된 특별약관의 부활(효력회복))부활(효력회복)되는 특별약관의 보장개시는 4-1. 반려견 '
 '의료비(치과및구강질환포함)(수\n'
 '술당일제외, 검사비포함)(재가입형) 특별약관 제22조(보험료의 납입을 연체하여 해지된\n'
 '특별약관의 부활(효력회복))를 따릅니다. 이 경우 부활(효력회복)일을 보험계약일로 하여'),
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
