from langchain_core.documents import Document

chunk = Document(
    page_content=('<지급보험금의 계산>\n'
 '{(피보험자가 부담한 1일당 의료비 - 1일당 자기부담금) × 보상비율}과 보험증권에 기재된 1일당 보상한도액 중 적은 금액\n'
 '<예시안내>\n'
 '[반려묘 의료비(치과및구강질환포함)(재가입형) 계산]\n'
 '· 보험가입금액 : 10만원, 보상비율 : 70%, 자기부담금 : 3만원 · 예시1\n'
 '- 피보험자가 부담한 1일당 의료비 22만원 - 보험금 지급금액\n'
 '= [(22만원 - 3만원) × 70%, 10만원] 중 적은 금액 = 10만원\n'
 '· 예시2\n'
 '- 피보험자가 부담한 1일당 의료비 13만원 - 보험금 지급금액'),
    metadata={'source_doc': {'total_pages': 129},
 'doc': {'doc_type': 'terms',
         'file_name': 'samsung_5_pet_majestic_cat_31_47_50_159.pdf',
         'insurer_code': 'samsung',
         'product_code': '5',
         'product_name': '(장기)무배당 삼성화재 펫보험 의기냥냥',
         'total_pages': 107,
         'page': 97},
 'term_type': 'special',
 'clause': {'clause_type': 'coverage', 'risk_domains': ['dental']},
 'indexing': {'chunk_id': 'chunk_000537',
              'chunk_char_len': 287,
              'embedding_model': 'solar-embedding-1-large',
              'tag_method': 'rule',
              'tag_confidence': 0.8}},
)
