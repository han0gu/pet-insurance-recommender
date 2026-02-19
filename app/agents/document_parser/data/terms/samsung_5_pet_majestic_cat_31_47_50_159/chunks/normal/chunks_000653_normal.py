from langchain_core.documents import Document

chunk = Document(
    page_content=('· 예시2\n'
 '- 피보험자가 부담한 수술 당일 의료비 290만원 - 4-1. 반려묘 의료비(치과및구강질환포함)(재가입형) 특별약관에서 지급한 보험금 : '
 '10만원 - 보험금 지급금액\n'
 '= [(290만원 - 10만원) × 70%, 200만원] 중 적은 금액 = 196만원'),
    metadata={'source_doc': {'total_pages': 129},
 'doc': {'doc_type': 'terms',
         'file_name': 'samsung_5_pet_majestic_cat_31_47_50_159.pdf',
         'insurer_code': 'samsung',
         'product_code': '5',
         'product_name': '(장기)무배당 삼성화재 펫보험 의기냥냥',
         'total_pages': 107,
         'page': 107},
 'term_type': 'special',
 'clause': {'clause_type': 'renewal', 'risk_domains': ['dental']},
 'indexing': {'chunk_id': 'chunk_000653',
              'chunk_char_len': 146,
              'embedding_model': 'solar-embedding-1-large',
              'tag_method': 'rule',
              'tag_confidence': 0.55}},
)
