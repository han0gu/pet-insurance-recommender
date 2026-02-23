from langchain_core.documents import Document

chunk = Document(
    page_content=('- 니다.\n'
 '- 4. 한국표준질병∙사인분류 : 제9차 개정 한국표준질병·사인분류(통계청 고시 제2025-\n'
 '- 299호, 2026. 1. 1 시행)를 말하며 이후 한국표준질병·사인분류가 개정되는 경우는\n'
 '- 개정된 기준에 따라 이 약관에서 보장하는 질병(상병) 해당 여부를 판단합니다.\n'
 '# ③ 지급금과 이자율 관련 용어1. 연단위 복리: 회사가 지급할 금전에 이자를 줄 때 1년마다 마지막 날에 그 이자를'),
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
