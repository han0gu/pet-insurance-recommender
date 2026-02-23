from langchain_core.documents import Document

chunk = Document(
    page_content=('| 37 | 오른쪽 어깨 |\n'
 '| 38 | 왼팔(왼쪽 어깨 제외, 왼손 포함) |\n'
 '| 39 | 오른팔(오른쪽 어깨 제외, 오른손 포함) |\n'
 '| 40 | 왼손(왼쪽 손목 관절 이하) |\n'
 '| 41 | 오른손(오른쪽 손목 관절 이하) |\n'
 '| 42 | 왼쪽 고관절 |\n'
 '| 43 | 오른쪽 고관절 |\n'
 '| 44 | 왼쪽 다리(왼쪽 고관절 제외, 왼발 포함) |\n'
 '| 45 | 오른쪽 다리(오른쪽 고관절 제외, 오른발 포함) |\n'
 '| 46 | 왼발(왼쪽 발목 관절 이하) |\n'
 '| 47 | 오른발(오른쪽 발목 관절 이하) |'),
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
