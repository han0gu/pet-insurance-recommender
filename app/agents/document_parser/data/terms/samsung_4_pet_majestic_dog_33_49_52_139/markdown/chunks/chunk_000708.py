from langchain_core.documents import Document

chunk = Document(
    page_content=('우 보험료 납입을 면제하여 드리지 않습니다.제3조 (보험료의 납입을 연체하여 해지된 특별약관의 부활(효력회복))회사는 이 특별약관의 '
 '부활(효력회복)청약을 받은 경우에는 계약의 부활(효력회복)을 승\n'
 '낙한 경우에 한하여 보험계약 「보험료의 납입을 연체하여 해지된 특별약관의 부활(효력'),
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
