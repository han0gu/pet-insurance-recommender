from langchain_core.documents import Document

chunk = Document(
    page_content=("id='8' data-category='paragraph' style='font-size:16px'>\uf000 계약자는 이 계약의 "
 '해약환급금 범위 내에서 회사가 정<br>한 방법에 따라 대출(이하「보험계약대출」이라 합니다)을<br>받을 수 있습니다'),
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
 'indexing': {'chunk_id': 'chunk_000226',
              'chunk_char_len': 134,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'simple',
              'tag_method': 'rule',
              'tag_confidence': 0.25}},
)
