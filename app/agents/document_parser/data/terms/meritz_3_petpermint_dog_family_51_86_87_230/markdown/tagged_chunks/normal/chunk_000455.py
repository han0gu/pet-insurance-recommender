from langchain_core.documents import Document

chunk = Document(
    page_content=('상비율 50% 가입, 연간 첫번째 MRI,CT 및 내시경처\n'
 '치)- ·피보험자가 부담한 치료비 103만원\n'
 '- ·보험금 지급금액\n'
 '= [(103만원 - 3만원)×50%, 30만원] 중 적은금액\n'
 '= 30만원③ 입원 중 MRI,CT 및 내시경처치와 수술을 동시에 한\n'
 '경우(보상비율 50% 가입)- ·피보험자가 부담한 수술당일 치료비 410만원\n'
 '- ·보험금 지급금액\n'
 '- = [(410만원-3만원)×50%, 200만원] 중 적은금액\n'
 '- = 200만원(MRI,CT 및 내시경처치와 수술을 동시에\n'
 '- 하더라도 수술한도로 지급)'),
    metadata={'source_doc': {'total_pages': 180},
 'doc': {'doc_type': 'terms',
         'file_name': 'meritz_3_petpermint_dog_family_51_86_87_230.pdf',
         'insurer_code': 'meritz',
         'product_code': '3',
         'product_name': '무배당 펫퍼민트 Puppy&Family보험 다이렉트2601',
         'total_pages': 1,
         'page': 1},
 'term_type': 'unknown',
 'clause': {'clause_type': 'limit', 'risk_domains': ['other']},
 'indexing': {'chunk_id': 'chunk_000455',
              'chunk_char_len': 285,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'normal',
              'tag_method': 'rule',
              'tag_confidence': 0.8}},
)
