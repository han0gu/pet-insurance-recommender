from langchain_core.documents import Document

chunk = Document(
    page_content=('- 하지 않는다.\n'
 '- 3) 손가락에는 첫째 손가락에 2개의 손가락관절이 있다.\n'
 '- 그중 심장에서 가까운 쪽부터 중수지관절, 지관절이\n'
 '- 라 한다.\n'
 '- 4) 다른 네 손가락에는 3개의 손가락관절이 있다. 그\n'
 '- 중 심장에서 가까운 쪽부터 중수지관절, 제1지관절\n'
 '- (근위지관절) 및 제2지관절(원위지관절)이라 부른다.\n'
 '- 5) “손가락을 잃었을 때”라 함은 첫째 손가락에서는 지\n'
 '- 관절부터 심장에서 가까운 쪽에서, 다른 네 손가락에서\n'
 '- 는 제1지관절(근위지관절)부터(제1지관절 포함) 심장'),
    metadata={'source_doc': {'total_pages': 160},
 'doc': {'doc_type': 'terms',
         'file_name': 'meritz_2_petpermint_cat_family_45_82_83_206.pdf',
         'insurer_code': 'meritz',
         'product_code': '2',
         'product_name': '무배당 펫퍼민트 Cat&Family보험 다이렉트2601',
         'total_pages': 1,
         'page': 1},
 'term_type': 'unknown',
 'clause': {'clause_type': 'other', 'risk_domains': ['digestive', 'joint']},
 'indexing': {'chunk_id': 'chunk_000591',
              'chunk_char_len': 274,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'normal',
              'tag_method': 'rule',
              'tag_confidence': 0.25}},
)
