from langchain_core.documents import Document

chunk = Document(
    page_content=('· 예시1- \n'
 '- - 피보험자가 부담한 1일당 의료비 22만원\n'
 '- - 보험금 지급금액\n'
 '= [(22만원 - 3만원) × 70%, 10만원] 중 적은 금액\n'
 '= 10만원# · 예시2- - 피보험자가 부담한 1일당 의료비 13만원\n'
 '- - 보험금 지급금액\n'
 '- = [(13만원 - 3만원) × 70%, 10만원] 중 적은 금액\n'
 '- = 7만원\n'
 '- ⑤ 제1항의 「연간」 이란 계약일로부터 매1년 단위로 도래하는 계약해당일 전일까지의\n'
 '- 기간을 의미합니다.\n'
 '- ⑥ 제2항에도 불구하고 제27조 (특별약관의 재가입에 관한 사항) 제1항 및 제2항에 따라'),
    metadata={'source_doc': {'total_pages': 129},
 'doc': {'doc_type': 'terms',
         'file_name': 'samsung_5_pet_majestic_cat_31_47_50_159.pdf',
         'insurer_code': 'samsung',
         'product_code': '5',
         'product_name': '(장기)무배당 삼성화재 펫보험 의기냥냥',
         'total_pages': 1,
         'page': 1},
 'term_type': 'unknown',
 'clause': {'clause_type': 'coverage', 'risk_domains': ['digestive']},
 'indexing': {'chunk_id': 'chunk_000458',
              'chunk_char_len': 299,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'simple',
              'tag_method': 'rule',
              'tag_confidence': 0.8}},
)
