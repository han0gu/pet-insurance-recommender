from langchain_core.documents import Document

chunk = Document(
    page_content=('100개 이상의 병상 구비, 병상수에 따라 일정 개수의 진료과목을 갖추고, 각 진료과목마다 전속하\n'
 '는 전문의를 둔 병원을 말합니다.# 제3조 ( 「깁스(Cast)치료」 의 정의)① 이 특별약관에서 「깁스(Cast) 치료」 라 함은 병원 '
 '또는 의원의 의사의 면허를 가진 자\n'
 '(이하 「의사」 라 합니다)가 치료를 직접적인 목적으로 「깁스(Cast)치료」 가 필요하다고 인정되는 경우로서 병원에서 의사의 관리하에 '
 '석고붕대 또는 섬유유리붕대'),
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
