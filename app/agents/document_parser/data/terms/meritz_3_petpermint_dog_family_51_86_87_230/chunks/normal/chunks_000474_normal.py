from langchain_core.documents import Document

chunk = Document(
    page_content=('【보험금 지급금액 산출방식】\n'
 '보험금 지급금액 = [(피보험자가 부담한 치료비－자기부담금) × 보상비율, 지급 한도액] 중 적은 금액\n'
 '【보험금 지급금액[자기부담금 3만원 예시]】\n'
 '① 통원 중 수술을 하지 않은 경우(보상비율 70% 가입, MRI,CT 및 내시경처치를 받지 않은 날)\n'
 '·피보험자가 부담한 치료비 33만원 ·보험금 지급금액\n'
 '= [(33만원 - 3만원)×70%, 30만원] 중 적은금액 = 21만원\n'
 '② 통원 중 MRI,CT 및 내시경처치를 받은 날의 경우(보 상비율 70% 가입, 연간 첫번째 MRI,CT 및 내시경처 치)'),
    metadata={'source_doc': {'total_pages': 180},
 'doc': {'doc_type': 'terms',
         'file_name': 'meritz_3_petpermint_dog_family_51_86_87_230.pdf',
         'insurer_code': 'meritz',
         'product_code': '3',
         'product_name': '무배당 펫퍼민트 Puppy&Family보험 다이렉트2601',
         'total_pages': 180,
         'page': 149},
 'term_type': 'special',
 'clause': {'clause_type': 'deductible', 'risk_domains': ['other']},
 'indexing': {'chunk_id': 'chunk_000474',
              'chunk_char_len': 297,
              'embedding_model': 'solar-embedding-1-large',
              'tag_method': 'rule',
              'tag_confidence': 0.85}},
)
