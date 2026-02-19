from langchain_core.documents import Document

chunk = Document(
    page_content=('<지급보험금의 계산>\n'
 '{ ( 피보험자가 부담한 MRI또는 CT촬영 당일 의료비\n'
 '- 4-1. 반려묘 의료비(치과및구강질환포함)(재가입형) 특별약관 지급한 보험금 - 4-2. 반려묘 수술비(치과및구강질환포함) '
 '확대보장(재가입형) 추가특별약관 지급한 보험금 - 자기부담금 ) × 보상비율 }\n'
 '과 보험증권에서 정한 1일당 보상한도액 중 적은 금액\n'
 '<예시안내>\n'
 '[반려묘 의료비 확대보장(MRI.CT)(연간1회한(재가입형) 계산]'),
    metadata={'source_doc': {'total_pages': 129},
 'doc': {'doc_type': 'terms',
         'file_name': 'samsung_5_pet_majestic_cat_31_47_50_159.pdf',
         'insurer_code': 'samsung',
         'product_code': '5',
         'product_name': '(장기)무배당 삼성화재 펫보험 의기냥냥',
         'total_pages': 107,
         'page': 117},
 'term_type': 'special',
 'clause': {'clause_type': 'coverage', 'risk_domains': ['dental', 'digestive']},
 'indexing': {'chunk_id': 'chunk_000743',
              'chunk_char_len': 234,
              'embedding_model': 'solar-embedding-1-large',
              'tag_method': 'rule',
              'tag_confidence': 0.8}},
)
