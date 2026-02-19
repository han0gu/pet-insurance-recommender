from langchain_core.documents import Document

chunk = Document(
    page_content=('·피보험자가 부담한 치료비 23만원 ·보험금 지급금액\n'
 '= [(23만원 - 3만원)×70%, 15만원] 중 적은금액 = 14만원\n'
 '② 통원 중 MRI,CT 및 내시경처치를 받은 날의 경우(보 상비율 70% 가입, 연간 첫번째 MRI,CT 및 내시경처 치)\n'
 '·피보험자가 부담한 치료비 103만원 ·보험금 지급금액\n'
 '= [(103만원 - 3만원)×70%, 50만원] 중 적은금액 = 50만원\n'
 '③ 통원 중 MRI,CT 및 내시경처치와 수술을 동시에 한 경우(보상비율 70% 가입)'),
    metadata={'source_doc': {'total_pages': 160},
 'doc': {'doc_type': 'terms',
         'file_name': 'meritz_2_petpermint_cat_family_45_82_83_206.pdf',
         'insurer_code': 'meritz',
         'product_code': '2',
         'product_name': '무배당 펫퍼민트 Puppy&Family보험 다이렉트2601',
         'total_pages': 160,
         'page': 145},
 'term_type': 'special',
 'clause': {'clause_type': 'coverage', 'risk_domains': ['other']},
 'indexing': {'chunk_id': 'chunk_000479',
              'chunk_char_len': 262,
              'embedding_model': 'solar-embedding-1-large',
              'tag_method': 'rule',
              'tag_confidence': 0.8}},
)
