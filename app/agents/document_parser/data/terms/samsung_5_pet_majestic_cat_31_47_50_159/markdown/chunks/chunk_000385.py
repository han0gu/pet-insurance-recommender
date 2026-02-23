from langchain_core.documents import Document

chunk = Document(
    page_content=('| 강 간 | 강 간 | 500만원 |\n'
 '| 강 도 | 강 도 | 100만원 |\n'
 '| 상해, 폭행 및 폭력 (예상치료기간별) | 전치 6개월 초과 | 300만원 |\n'
 '| 상해, 폭행 및 폭력 (예상치료기간별) | 전치 4개월 초과 6개월 이하 | 200만원 |\n'
 '| 상해, 폭행 및 폭력 (예상치료기간별) | 전치 2개월 초과 4개월 이하 | 150만원 |\n'
 '| 상해, 폭행 및 폭력 (예상치료기간별) | 전치 1개월 초과 2개월 이하 | 100만원 |\n'
 '- ② 제1항에서 「상해, 폭행 및 폭력」의 예상치료기간은 관할 검·경찰 기관에 피해 입'),
    metadata={'source_doc': {'total_pages': 129},
 'doc': {'doc_type': 'terms',
         'file_name': 'samsung_5_pet_majestic_cat_31_47_50_159.pdf',
         'insurer_code': 'samsung',
         'product_code': '5',
         'product_name': '(장기)무배당 삼성화재 펫보험 의기냥냥',
         'total_pages': 1,
         'page': 1},
 'term_type': 'unknown'},
)
