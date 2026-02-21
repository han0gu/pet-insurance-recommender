from langchain_core.documents import Document

chunk = Document(
    page_content=('- 반려묘 의료비 확대보장(이물제거 특정처치](연간2회한)(재가입형) 200만원\n'
 '- · 보상비율 : 70%, 자기부담금 : 3만원\n'
 '- · 수술여부 : 수술을 하지 않은 날의 경우\n'
 '# · 예사1(이물제거(내시경)보험금 지급금액 예시)- 피보험자가 부담한 이물제거(내시경) 치료 당일 의료비 213만원- 4-1. 반려묘 '
 '의료비(치과및구강질환포함)(재가입형) 특별약관에서 지급한 보험금 : 10만원\n'
 '- 보험금 지급금액- = [(213만원 - 10만원 - 3만원) × 70%, 200만원] 중 적은 금액\n'
 '- = 140만원'),
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
 'indexing': {'chunk_id': 'chunk_000593',
              'chunk_char_len': 287,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'simple',
              'tag_method': 'rule',
              'tag_confidence': 0.8}},
)
