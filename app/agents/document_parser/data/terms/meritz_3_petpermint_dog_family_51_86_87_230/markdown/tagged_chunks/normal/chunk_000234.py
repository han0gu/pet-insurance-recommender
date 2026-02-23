from langchain_core.documents import Document

chunk = Document(
    page_content=('| 통원 의료비 | 통원 중 수술을 한 날의 경우 | 1일당 3만원/5만원 중 보험증권에 기재된 자기부담금 | 수술당일에 한하여 1일당 '
 '200만원 |\n'
 '【보험금 지급금액 산출방식】보험금 지급금액 = [(피보험자가 부담한 치료비－자기부담금)\n'
 '× 보상비율, 지급 한도액] 중 적은 금액【보험금 지급금액(자기부담금 3만원인 경우)[예시]】① 통원 중 수술을 하지 않은 경우(보상비율 '
 '70%)- ·피보험자가 부담한 치료비 23만원\n'
 '- ·보험금 지급금액\n'
 '- = [(23만원 - 3만원)×70%, 15만원] 중 적은금액\n'
 '- = 14만원'),
    metadata={'source_doc': {'total_pages': 180},
 'doc': {'doc_type': 'terms',
         'file_name': 'meritz_3_petpermint_dog_family_51_86_87_230.pdf',
         'insurer_code': 'meritz',
         'product_code': '3',
         'product_name': '무배당 펫퍼민트 Puppy&Family보험 다이렉트2601',
         'total_pages': 1,
         'page': 1},
 'term_type': 'unknown',
 'clause': {'clause_type': 'deductible', 'risk_domains': ['other']},
 'indexing': {'chunk_id': 'chunk_000234',
              'chunk_char_len': 292,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'normal',
              'tag_method': 'rule',
              'tag_confidence': 0.85}},
)
