from langchain_core.documents import Document

chunk = Document(
    page_content=('- = 140만원\n'
 '# · 예시2,이물제거(구토유도약물)보험금 지급금액 예시)- 피보험자가 부담한 이물제거(구토유도약물) 치료 당일 의료비 53만원\n'
 '- 4-1. 반려묘 의료비(치과및구강질환포함)(재가입형) 특별약관에서 지급한 보험금 : 10만원\n'
 '- 보험금 지급금액- \n'
 '- = [(53만원 - 10만원 - 3만원) × 70%, 20만원] 중 적은 금액\n'
 '- = 20만원\n'
 '- ④ 제3항의 「자기부담금」 이란 보험증권에 기재된 4-1. 반려묘 의료비(치과및구강질환\n'
 '- 포함)(재가입형) 특별약관의 자기부담금을 말합니다'),
    metadata={'source_doc': {'total_pages': 129},
 'doc': {'doc_type': 'terms',
         'file_name': 'samsung_5_pet_majestic_cat_31_47_50_159.pdf',
         'insurer_code': 'samsung',
         'product_code': '5',
         'product_name': '(장기)무배당 삼성화재 펫보험 의기냥냥',
         'total_pages': 1,
         'page': 1},
 'term_type': 'unknown',
 'clause': {'clause_type': 'deductible',
            'risk_domains': ['dental', 'digestive']},
 'indexing': {'chunk_id': 'chunk_000594',
              'chunk_char_len': 284,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'normal',
              'tag_method': 'rule',
              'tag_confidence': 0.85}},
)
