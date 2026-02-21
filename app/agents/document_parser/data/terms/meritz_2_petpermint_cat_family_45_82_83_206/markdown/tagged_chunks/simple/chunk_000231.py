from langchain_core.documents import Document

chunk = Document(
    page_content=('- 에 따른 신고를 한 진료기관을 말한다.\n'
 '\uf000 회사가 보상하는 비용은 각 항목별 피보험자가 부담한\n'
 '치료비에서 보험증권에 기재된 자기부담금을 각각 차감한\n'
 '후, 보험증권에 기재된 보상비율(70%)을 곱한 금액을 아래\n'
 '에서 정한 금액을 한도로 보상합니다.| 항목 | 항목 | 자기부담금 | 지급 한도 |\n'
 '| --- | --- | --- | --- |\n'
 '| 통원 의료비 | 통원 중 수술을 하지 않은 날의 경우 | 1일당 3만원/5만원 중 보험증권에 기재된 자기부담금 | 1일당 15만원 |'),
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
 'indexing': {'chunk_id': 'chunk_000231',
              'chunk_char_len': 269,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'simple',
              'tag_method': 'rule',
              'tag_confidence': 0.8}},
)
