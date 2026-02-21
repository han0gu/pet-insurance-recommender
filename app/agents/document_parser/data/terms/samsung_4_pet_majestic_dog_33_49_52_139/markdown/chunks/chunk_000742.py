from langchain_core.documents import Document

chunk = Document(
    page_content=('| 47 | 오른발(오른쪽 발목 관절 이하) |\n'
 '| 48 | 상 ·하악골(위·아래턱뼈) |\n'
 '| 49 | 쇄골 |\n'
 '| 50 | 늑골(갈비뼈) |\n'
 '# 【 붙임2】 특정질병 분류표약관에 규정하는 특정질병으로 분류되는 질병은 제9차 개정 한국표준질병·사인분류(통계\n'
 '청 고시 제2025-299호, 2026. 1. 1 시행) 중 다음에 적은 질병을 말합니다.| 구분 | 대 상 질 병 | 분 류 번 호 | '
 '분 류 번 호 |\n'
 '| --- | --- | --- | --- |\n'
 '| 51 | 담석증 | K80 | 담석증 |'),
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
