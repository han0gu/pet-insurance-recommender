from langchain_core.documents import Document

chunk = Document(
    page_content=('# <지급금액 예시># ·계약일 : 2026년 1월 1일| <1> SB030(안면부 이외,근육) 2회 실시 ↓ | <1> '
 'SB030(안면부 이외,근육) 2회 실시 ↓ | <2> SA027(안면부,변연절제, 3cm이상) 3회 실시(1일 1회) ↓ |\n'
 '| --- | --- | --- |\n'
 '| 2026.1.1 | 2026.3.1 | 2026.8.1 ~ 2026.8.3 2026.12.31 |\n'
 '| <1> | · (B) 보험금 1회 |\n'
 '| --- | --- |'),
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
