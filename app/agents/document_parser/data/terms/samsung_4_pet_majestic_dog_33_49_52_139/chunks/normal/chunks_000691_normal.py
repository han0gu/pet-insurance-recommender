from langchain_core.documents import Document

chunk = Document(
    page_content=('∙ 예시2\n'
 '- 피보험자가 부담한 수술당일 의료비 303만원 - 보험금 지급금액\n'
 '= [(303만원 - 3만원) × 70%, 200만원] 중 적은 금액 = 200만원\n'
 '⑥ 제1항의 「연간」이란 계약일로부터 매1년 단위로 도래하는 계약해당일 전일까지의 기간을 의미합니다. ⑦ 제2항에도 불구하고 4-1. '
 '반려견 의료비(치과및구강질환포함)(수술당일제외, 검사비포'),
    metadata={'source_doc': {'total_pages': 107},
 'doc': {'doc_type': 'terms',
         'file_name': 'samsung_4_pet_majestic_dog_33_49_52_139.pdf',
         'insurer_code': 'samsung',
         'product_code': '4',
         'product_name': '(장기)무배당 삼성화재 펫보험 위풍댕댕',
         'total_pages': 91,
         'page': 114},
 'term_type': 'special',
 'clause': {'clause_type': 'exclusion',
            'risk_domains': ['dental', 'digestive']},
 'indexing': {'chunk_id': 'chunk_000691',
              'chunk_char_len': 197,
              'embedding_model': 'solar-embedding-1-large',
              'tag_method': 'rule',
              'tag_confidence': 0.85}},
)
