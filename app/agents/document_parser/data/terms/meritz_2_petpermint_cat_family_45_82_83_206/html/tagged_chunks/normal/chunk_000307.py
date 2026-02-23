from langchain_core.documents import Document

chunk = Document(
    page_content=("id='39' data-category='paragraph' style='font-size:20px'>\uf000 보험수익자와 회사가 "
 '보험금 지급사유에 대해 합의하지<br>못할 때는 보험수익자와 회사가 함께 제3자를 정하고 그 제<br>3자의 의견에 따를 수 있습니다'),
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
 'indexing': {'chunk_id': 'chunk_000307',
              'chunk_char_len': 144,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'normal',
              'tag_method': 'rule',
              'tag_confidence': 0.55}},
)
