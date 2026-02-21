from langchain_core.documents import Document

chunk = Document(
    page_content=('| 통 원 의 료 비 Ⅲ | 통원 중 수술을 한 날의 경우 | 통원 중 수술을 한 날의 경우 | 통원 중 수술을 한 날의 경우 | 1일당 '
 '3만원/ 5만원 중 보험증 권에 기재된 자기부 담금 | 수술당일에 한하여 1일당 200만원 |\n'
 '147# 【보험금 지급금액 산출방식】보험금 지급금액 = [(피보험자가 부담한 치료비－자기부담금)\n'
 '× 보상비율, 지급 한도액] 중 적은 금액【보험금 지급금액[자기부담금 3만원 예시]】① 통원 중 수술을 하지 않은 경우(보상비율 50% '
 '가입,'),
    metadata={'source_doc': {'total_pages': 160},
 'doc': {'doc_type': 'terms',
         'file_name': 'meritz_2_petpermint_cat_family_45_82_83_206.pdf',
         'insurer_code': 'meritz',
         'product_code': '2',
         'product_name': '무배당 펫퍼민트 Cat&Family보험 다이렉트2601',
         'total_pages': 1,
         'page': 1},
 'term_type': 'unknown',
 'clause': {'clause_type': 'coverage', 'risk_domains': ['other']},
 'indexing': {'chunk_id': 'chunk_000388',
              'chunk_char_len': 263,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'simple',
              'tag_method': 'rule',
              'tag_confidence': 0.8}},
)
