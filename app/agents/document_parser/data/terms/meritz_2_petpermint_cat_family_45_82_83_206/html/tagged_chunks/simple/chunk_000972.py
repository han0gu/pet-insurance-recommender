from langchain_core.documents import Document

chunk = Document(
    page_content=("결손</h1><br><h1 id='63' style='font-size:20px'>3) 목</h1><br><h1 id='64' "
 "style='font-size:20px'>손바닥 크기 이상의 추상(추한 모습)</h1><h1 id='65' "
 "style='font-size:20px'>라"),
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
 'indexing': {'chunk_id': 'chunk_000972',
              'chunk_char_len': 155,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'simple',
              'tag_method': 'rule',
              'tag_confidence': 0.25}},
)
