from langchain_core.documents import Document

chunk = Document(
    page_content=('(서혜부 탈장 포함)</td></tr><tr><td>KEA005</td><td>회음부 '
 '탈장</td></tr><tr><td>KEA006</td><td>대퇴 '
 '탈장</td></tr><tr><td>KEA007</td><td>직장탈장</td></tr><tr><td>KEA008</td><td>기타 '
 '복부탈장</td></tr><tr><td>KFA001</td><td>복막염</td></tr><tr><td>KGA001</td><td>트리코모나스증</td></tr><tr><td>KGA002</td><td>지아르디아'),
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
 'indexing': {'chunk_id': 'chunk_000888',
              'chunk_char_len': 279,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'simple',
              'tag_method': 'rule',
              'tag_confidence': 0.25}},
)
