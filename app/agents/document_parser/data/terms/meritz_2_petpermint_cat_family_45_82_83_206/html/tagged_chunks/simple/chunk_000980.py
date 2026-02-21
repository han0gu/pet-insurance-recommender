from langchain_core.documents import Document

chunk = Document(
    page_content=('때</td><td>10</td></tr><tr><td>4) 척추(등뼈)에 심한 기형을 남긴 '
 '때</td><td>50</td></tr><tr><td>5) 척추(등뼈)에 뚜렷한 기형을 남긴 '
 '때</td><td>30</td></tr><tr><td>6) 척추(등뼈)에 약간의 기형을 남긴 '
 '때</td><td>15</td></tr><tr><td>7) 추간판탈출증으로 인한 심한 신경 '
 '장해</td><td>20</td></tr><tr><td>8) 추간판탈출증으로 인한 뚜렷한 신경 '
 '장해</td><td>15</td></tr><tr><td>9)'),
    metadata={'source_doc': {'total_pages': 160},
 'doc': {'doc_type': 'terms',
         'file_name': 'meritz_2_petpermint_cat_family_45_82_83_206.pdf',
         'insurer_code': 'meritz',
         'product_code': '2',
         'product_name': '무배당 펫퍼민트 Cat&Family보험 다이렉트2601',
         'total_pages': 1,
         'page': 1},
 'term_type': 'unknown',
 'clause': {'clause_type': 'other', 'risk_domains': ['digestive', 'head']},
 'indexing': {'chunk_id': 'chunk_000980',
              'chunk_char_len': 292,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'simple',
              'tag_method': 'rule',
              'tag_confidence': 0.25}},
)
