from langchain_core.documents import Document

chunk = Document(
    page_content=('(유리체 변성 포함)</td></tr><tr><td>FGA001</td><td>녹내장 '
 '(좌안)</td></tr><tr><td>FGA002</td><td>녹내장 '
 '(우안)</td></tr><tr><td>FGA003</td><td>동양안충증</td></tr><tr><td>FGA004</td><td>기타 '
 '안과'),
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
 'indexing': {'chunk_id': 'chunk_000858',
              'chunk_char_len': 169,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'simple',
              'tag_method': 'rule',
              'tag_confidence': 0.25}},
)
