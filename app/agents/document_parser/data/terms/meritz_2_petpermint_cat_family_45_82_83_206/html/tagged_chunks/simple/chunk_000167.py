from langchain_core.documents import Document

chunk = Document(
    page_content=("id='23' style='font-size:20px'>제24조(보험나이 등)</h1><br><p id='24' "
 "data-category='paragraph' style='font-size:16px'>\uf000 이 약관에서의 피보험자의 나이는 "
 '보험나이를 기준으로<br>합니다'),
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
 'indexing': {'chunk_id': 'chunk_000167',
              'chunk_char_len': 149,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'simple',
              'tag_method': 'rule',
              'tag_confidence': 0.25}},
)
