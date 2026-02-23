from langchain_core.documents import Document

chunk = Document(
    page_content=('108# 【보험금 지급금액 산출방식】보험금 지급금액 = [(피보험자가 부담한 치료비－자기부담금)\n'
 '× 보상비율, 지급 한도액] 중 적은 금액【보험금 지급금액(자기부담금 3만원인 경우)[예시]】① 통원 중 수술을 하지 않은 경우(보상비율 '
 '70%)- ·피보험자가 부담한 치료비 33만원\n'
 '- ·보험금 지급금액\n'
 '= [(33만원 - 3만원)×70%, 30만원] 중 적은금액\n'
 '= 21만원② 통원 중 수술을 한 경우(보상비율 70%)- ·피보험자가 부담한 수술당일 치료비 400만원\n'
 '- ·보험금 지급금액'),
    metadata={'source_doc': {'total_pages': 180},
 'doc': {'doc_type': 'terms',
         'file_name': 'meritz_3_petpermint_dog_family_51_86_87_230.pdf',
         'insurer_code': 'meritz',
         'product_code': '3',
         'product_name': '무배당 펫퍼민트 Puppy&Family보험 다이렉트2601',
         'total_pages': 1,
         'page': 1},
 'term_type': 'unknown',
 'clause': {'clause_type': 'coverage', 'risk_domains': ['other']},
 'indexing': {'chunk_id': 'chunk_000226',
              'chunk_char_len': 273,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'simple',
              'tag_method': 'rule',
              'tag_confidence': 0.8}},
)
