from langchain_core.documents import Document

chunk = Document(
    page_content=('청약을 받고, 제1회 보험료를 받<br>은 경우에 건강진단을 받지 않는 계약은 청약일, 진단계약<br>은 진단일(재진단의 경우에는 최종 '
 '진단일)부터 30일 이내<br>에 승낙 또는 거절하여야 하며, 승낙한 때에는 보험증권을<br>드립니다'),
    metadata={'source_doc': {'total_pages': 160},
 'doc': {'doc_type': 'terms',
         'file_name': 'meritz_2_petpermint_cat_family_45_82_83_206.pdf',
         'insurer_code': 'meritz',
         'product_code': '2',
         'product_name': '무배당 펫퍼민트 Cat&Family보험 다이렉트2601',
         'total_pages': 1,
         'page': 1},
 'term_type': 'unknown',
 'clause': {'clause_type': 'other', 'risk_domains': ['other']},
 'indexing': {'chunk_id': 'chunk_000345',
              'chunk_char_len': 132,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'normal',
              'tag_method': 'rule',
              'tag_confidence': 0.25}},
)
