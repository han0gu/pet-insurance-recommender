from langchain_core.documents import Document

chunk = Document(
    page_content=('. 다발<br>성늑골 기형의 경우 각각의 각(角) 변형을 합산하지 않<br>고 그 중 가장 높은 각(角) 변형을 기준으로 '
 "평가한다.</p><footer id='21' style='font-size:14px'>189</footer><table id='22' "
 'style=\'font-size:20px\'><thead></thead><tbody><tr><td><figure><img alt="" '
 'data-coord="top-left:(322,178); bottom-right:(913,577)"'),
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
 'indexing': {'chunk_id': 'chunk_001007',
              'chunk_char_len': 271,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'normal',
              'tag_method': 'rule',
              'tag_confidence': 0.25}},
)
