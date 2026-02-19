from langchain_core.documents import Document

chunk = Document(
    page_content=('·피보험자가 부담한 치료비 103만원 ·보험금 지급금액\n'
 '= [(103만원 - 3만원)×70%, 50만원] 중 적은금액 = 50만원\n'
 '③ 입원 중 MRI,CT 및 내시경처치와 수술을 동시에 한 경우(보상비율 70% 가입)\n'
 '·피보험자가 부담한 수술당일 치료비 410만원 ·보험금 지급금액\n'
 '= [(410만원-3만원)×70%, 250만원] 중 적은금액 = 250만원(MRI,CT 및 내시경처치와 수술을 동시에 하더라도 수술한도로 '
 '지급)'),
    metadata={'source_doc': {'total_pages': 180},
 'doc': {'doc_type': 'terms',
         'file_name': 'meritz_3_petpermint_dog_family_51_86_87_230.pdf',
         'insurer_code': 'meritz',
         'product_code': '3',
         'product_name': '무배당 펫퍼민트 Puppy&Family보험 다이렉트2601',
         'total_pages': 180,
         'page': 165},
 'term_type': 'special',
 'clause': {'clause_type': 'coverage', 'risk_domains': ['other']},
 'indexing': {'chunk_id': 'chunk_000541',
              'chunk_char_len': 238,
              'embedding_model': 'solar-embedding-1-large',
              'tag_method': 'rule',
              'tag_confidence': 0.8}},
)
