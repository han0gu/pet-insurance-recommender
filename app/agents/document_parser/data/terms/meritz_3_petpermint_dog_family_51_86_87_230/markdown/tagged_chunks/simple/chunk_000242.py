from langchain_core.documents import Document

chunk = Document(
    page_content=('후, 보험증권에 기재된 보상비율(50%)을 곱한 금액을 아래\n'
 '에서 정한 금액을 한도로 보상합니다.| 항목 | 항목 | 자기부담금 | 지급 한도 |\n'
 '| --- | --- | --- | --- |\n'
 '| 통원 의료비 | 통원 중 수술을 하지 않은 날의 경우 | 1일당 3만원/5만원 중 보험증권에 기재된 자기부담금 | 1일당 10만원 '
 '|\n'
 '| 통원 의료비 | 통원 중 수술을 한 날의 경우 | 1일당 3만원/5만원 중 보험증권에 기재된 자기부담금 | 수술당일에 한하여 1일당 '
 '150만원 |'),
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
 'indexing': {'chunk_id': 'chunk_000242',
              'chunk_char_len': 268,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'simple',
              'tag_method': 'rule',
              'tag_confidence': 0.8}},
)
