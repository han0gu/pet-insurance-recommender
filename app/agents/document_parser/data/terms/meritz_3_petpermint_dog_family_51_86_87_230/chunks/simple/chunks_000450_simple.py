from langchain_core.documents import Document

chunk = Document(
    page_content=('·피보험자가 부담한 치료비 13만원 ·보험금 지급금액\n'
 '= [(13만원 - 3만원)×50%, 10만원] 중 적은금액 = 5만원\n'
 '② 입원 중 수술을 한 경우(보상비율 50%)\n'
 '·피보험자가 부담한 수술당일 치료비 410만원 ·보험금 지급금액\n'
 '= [(410만원-3만원)×50%, 200만원] 중 적은금액 = 200만원'),
    metadata={'source_doc': {'total_pages': 180},
 'doc': {'doc_type': 'terms',
         'file_name': 'meritz_3_petpermint_dog_family_51_86_87_230.pdf',
         'insurer_code': 'meritz',
         'product_code': '3',
         'product_name': '무배당 펫퍼민트 Puppy&Family보험 다이렉트2601',
         'total_pages': 180,
         'page': 143},
 'term_type': 'special',
 'clause': {'clause_type': 'coverage', 'risk_domains': ['other']},
 'indexing': {'chunk_id': 'chunk_000450',
              'chunk_char_len': 173,
              'embedding_model': 'solar-embedding-1-large',
              'tag_method': 'rule',
              'tag_confidence': 0.8}},
)
