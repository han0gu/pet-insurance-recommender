from langchain_core.documents import Document

chunk = Document(
    page_content=('항목 | 자기부담금 | 지급 한도\n'
 '입원 의료비 Ⅱ | 입원 중 수술을 하지 않은 날의 경우 | 1일당 3만원/5만원 중 보험증권에 기재된 자기부담금 | 1일당 10만원\n'
 '입원 중 수술을 한 날의 경우 | 수술당일에 한하여 1일당 200만원\n'
 '【보험금 지급금액 산출방식】\n'
 '보험금 지급금액 = [(피보험자가 부담한 치료비－자기부담금) × 보상비율, 지급 한도액] 중 적은 금액\n'
 '【보험금 지급금액(자기부담금 3만원인 경우)[예시]】\n'
 '① 입원 중 수술을 하지 않은 경우(보상비율 50%)\n'
 '·피보험자가 부담한 치료비 13만원 ·보험금 지급금액'),
    metadata={'source_doc': {'total_pages': 180},
 'doc': {'doc_type': 'terms',
         'file_name': 'meritz_3_petpermint_dog_family_51_86_87_230.pdf',
         'insurer_code': 'meritz',
         'product_code': '3',
         'product_name': '무배당 펫퍼민트 Puppy&Family보험 다이렉트2601',
         'total_pages': 180,
         'page': 143},
 'term_type': 'special',
 'clause': {'clause_type': 'deductible', 'risk_domains': ['other']},
 'indexing': {'chunk_id': 'chunk_000449',
              'chunk_char_len': 296,
              'embedding_model': 'solar-embedding-1-large',
              'tag_method': 'rule',
              'tag_confidence': 0.85}},
)
