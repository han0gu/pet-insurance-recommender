from langchain_core.documents import Document

chunk = Document(
    page_content=('= 200만원· 예시2- 피보험자가 부담한 수술 당일 의료비 290만원\n'
 '- 4-1. 반려묘 의료비(치과및구강질환포함)(재가입형) 특별약관에서 지급한 보험금 : 10만원\n'
 '- 보험금 지급금액- = [(290만원 - 10만원) × 70%, 200만원] 중 적은 금액\n'
 '- = 196만원\n'
 '- ⑤ 제1항의 「연간」 이란 계약일로부터 매1년 단위로 도래하는 계약해당일 전일까지의\n'
 '- 기간을 의미합니다.\n'
 '- ⑥ 제2항에도 불구하고 4-1. 반려묘 의료비(치과및구강질환포함)(재가입형) 특별약관 제'),
    metadata={'source_doc': {'total_pages': 129},
 'doc': {'doc_type': 'terms',
         'file_name': 'samsung_5_pet_majestic_cat_31_47_50_159.pdf',
         'insurer_code': 'samsung',
         'product_code': '5',
         'product_name': '(장기)무배당 삼성화재 펫보험 의기냥냥',
         'total_pages': 1,
         'page': 1},
 'term_type': 'unknown',
 'clause': {'clause_type': 'coverage', 'risk_domains': ['dental', 'digestive']},
 'indexing': {'chunk_id': 'chunk_000550',
              'chunk_char_len': 269,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'simple',
              'tag_method': 'rule',
              'tag_confidence': 0.8}},
)
