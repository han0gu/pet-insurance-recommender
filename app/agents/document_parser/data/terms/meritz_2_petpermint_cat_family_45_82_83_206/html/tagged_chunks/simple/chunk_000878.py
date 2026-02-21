from langchain_core.documents import Document

chunk = Document(
    page_content=('사상균증 · 곰팡이성 피부염</td></tr><tr><td>LAA004</td><td>모낭염</td></tr><tr><td>LAA005 '
 'LAA006</td><td>모낭충증</td></tr><tr><td>LAA007</td><td>식이 알러지 알러지 피부염 (항원 '
 '특이적)</td></tr><tr><td>LAA008</td><td>아토피 (만성 '
 '피부염)</td></tr><tr><td>LAA009</td><td>지루성 '
 '피부염</td></tr><tr><td>LAA010</td><td>피하'),
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
 'indexing': {'chunk_id': 'chunk_000878',
              'chunk_char_len': 271,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'simple',
              'tag_method': 'rule',
              'tag_confidence': 0.25}},
)
