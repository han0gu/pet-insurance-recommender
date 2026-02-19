from langchain_core.documents import Document

chunk = Document(
    page_content=('④ 회사가 지급할 제1항에서 정한 보험금은 피보험자가 부담한 수술 당일 발생한 의료비 에서 4-1. 반려묘 '
 '의료비(치과및구강질환포함)(재가입형) 특별약관에서 지급한 보험금 을 차감한 후 보상비율을 곱한 금액이며 보험증권에 기재된 1일당 '
 '보상한도액을 한도 로 합니다.\n'
 '<지급보험금의 계산>\n'
 '{(피보험자가 부담한 수술 당일 의료비 -4-1. 반려묘 의료비(치과및구강질환포함)(재가입형) 특별 약관에서 지급한 보험금) × '
 '보상비율}과 보험증권에서 정한 1일당 보상한도액 중적은 금액\n'
 '<예시안내>'),
    metadata={'source_doc': {'total_pages': 129},
 'doc': {'doc_type': 'terms',
         'file_name': 'samsung_5_pet_majestic_cat_31_47_50_159.pdf',
         'insurer_code': 'samsung',
         'product_code': '5',
         'product_name': '(장기)무배당 삼성화재 펫보험 의기냥냥',
         'total_pages': 107,
         'page': 107},
 'term_type': 'special',
 'clause': {'clause_type': 'coverage', 'risk_domains': ['dental']},
 'indexing': {'chunk_id': 'chunk_000651',
              'chunk_char_len': 273,
              'embedding_model': 'solar-embedding-1-large',
              'tag_method': 'rule',
              'tag_confidence': 0.8}},
)
