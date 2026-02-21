from langchain_core.documents import Document

chunk = Document(
    page_content=("id='4' style='font-size:18px'>제14조(대표자의 지정)</h1><br><p id='5' "
 "data-category='paragraph' style='font-size:16px'>\uf000 계약자 또는 보험수익자가 2명 이상인 "
 '경우에는 각 대표<br>자를 1명 지정하여야 합니다'),
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
 'indexing': {'chunk_id': 'chunk_000073',
              'chunk_char_len': 162,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'normal',
              'tag_method': 'rule',
              'tag_confidence': 0.25}},
)
