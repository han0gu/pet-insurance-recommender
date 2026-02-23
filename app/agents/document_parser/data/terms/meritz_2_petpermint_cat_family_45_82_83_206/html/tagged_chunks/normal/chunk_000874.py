from langchain_core.documents import Document

chunk = Document(
    page_content=('(양성)</td></tr><tr><td>AFB013</td><td>상세미상의 피부 신생물 '
 '(악성)</td></tr><tr><td>AFC013</td><td>상세미상의 피부 신생물 (양성 또는 악성이 '
 '불확실한)</td></tr><tr><td>AFA014</td><td>기타 피부 신생물 '
 '(양성)</td></tr><tr><td>AFB014</td><td>기타 피부 신생물 '
 '(악성)</td></tr><tr><td>AFC014</td><td>기타 피부 신생물 (양성 또는 악성이 불확실'),
    metadata={'source_doc': {'total_pages': 160},
 'doc': {'doc_type': 'terms',
         'file_name': 'meritz_2_petpermint_cat_family_45_82_83_206.pdf',
         'insurer_code': 'meritz',
         'product_code': '2',
         'product_name': '무배당 펫퍼민트 Cat&Family보험 다이렉트2601',
         'total_pages': 1,
         'page': 1},
 'term_type': 'unknown',
 'clause': {'clause_type': 'other', 'risk_domains': ['skin']},
 'indexing': {'chunk_id': 'chunk_000874',
              'chunk_char_len': 267,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'normal',
              'tag_method': 'rule',
              'tag_confidence': 0.25}},
)
