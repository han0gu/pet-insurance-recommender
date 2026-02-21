from langchain_core.documents import Document

chunk = Document(
    page_content=('반려묘 수술비(치과및구강질환포함) 확대보장(재가입형) 200만원- · 보상비율 : 70%, 자기부담금 : 3만원\n'
 '- · 예시1\n'
 '- 피보험자가 부담한 수술 당일 의료비 310만원- - 4-1. 반려묘 의료비(치과및구강질환포함)(재가입형) 특별약관에서 지급한 보험금 '
 ': 10만원\n'
 '- - 보험금 지급금액\n'
 '= [(310만원 - 10만원) × 70%, 200만원] 중 적은 금액\n'
 '= 200만원· 예시2- 피보험자가 부담한 수술 당일 의료비 290만원'),
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
 'indexing': {'chunk_id': 'chunk_000549',
              'chunk_char_len': 246,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'simple',
              'tag_method': 'rule',
              'tag_confidence': 0.8}},
)
