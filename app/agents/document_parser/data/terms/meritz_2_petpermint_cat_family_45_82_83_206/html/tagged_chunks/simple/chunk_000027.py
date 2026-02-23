from langchain_core.documents import Document

chunk = Document(
    page_content=('. 그러나 그 후유장해가 이미 후유장해보험금을 지급받<br>은 동일한 부위에 가중된 때에는 최종 장해상태에 '
 '해당하는<br>후유장해보험금에서 이미 지급받은 후유장해보험금을 차감<br>하여 지급합니다'),
    metadata={'source_doc': {'total_pages': 160},
 'doc': {'doc_type': 'terms',
         'file_name': 'meritz_2_petpermint_cat_family_45_82_83_206.pdf',
         'insurer_code': 'meritz',
         'product_code': '2',
         'product_name': '무배당 펫퍼민트 Cat&Family보험 다이렉트2601',
         'total_pages': 1,
         'page': 1},
 'term_type': 'unknown',
 'clause': {'clause_type': 'other', 'risk_domains': ['digestive']},
 'indexing': {'chunk_id': 'chunk_000027',
              'chunk_char_len': 109,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'simple',
              'tag_method': 'rule',
              'tag_confidence': 0.25}},
)
