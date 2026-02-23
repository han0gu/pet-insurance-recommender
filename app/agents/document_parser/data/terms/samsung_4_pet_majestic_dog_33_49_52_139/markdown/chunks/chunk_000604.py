from langchain_core.documents import Document

chunk = Document(
    page_content=('| 파보바이러스 감염증 | 파보바이러스에 감염되어 구토와 설사 등의 증상을 일으킴 |\n'
 '| 디스템퍼바이러스 감염증 | 디스템퍼바이러스에 감염되어 호흡기 질환과 신경증상을 일으킴 |\n'
 '| 파라인플루엔자 감염증 | 파라인플루엔자에 감염되어, 기침, 가래, 콧물 등의 증상을 일으킴 |\n'
 '| 아데노바이러스 2형 감염증 | 아데노바이러스 2형 바이러스에 감염되어 호흡기 증상 등을 일으킴 |\n'
 '| 코로나바이러스 감염증 | 코로나바이러스성 장염으로 불리며, 소화계통의 바이러스 감염으로 인 해 구토, 설사 등의 증상을 일으킴 |'),
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
