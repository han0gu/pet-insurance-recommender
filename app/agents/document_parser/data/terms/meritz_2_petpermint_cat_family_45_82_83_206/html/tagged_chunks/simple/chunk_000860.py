from langchain_core.documents import Document

chunk = Document(
    page_content=('불명)</td></tr><tr><td>QBA003</td><td>눈 가려움증 (원인 불명)</td></tr><tr><td '
 'rowspan="19">3</td><td rowspan="19">순환기 질환</td><td>ACA001</td><td>순환기 계통의 양성 '
 '신생물</td></tr><tr><td>ACB001</td><td>순환기 계통의 악성 '
 '신생물</td></tr><tr><td>ACC001</td><td>순환기 계통의 신생물(양성 또는 악성이 불확'),
    metadata={'source_doc': {'total_pages': 160},
 'doc': {'doc_type': 'terms',
         'file_name': 'meritz_2_petpermint_cat_family_45_82_83_206.pdf',
         'insurer_code': 'meritz',
         'product_code': '2',
         'product_name': '무배당 펫퍼민트 Cat&Family보험 다이렉트2601',
         'total_pages': 1,
         'page': 1},
 'term_type': 'unknown',
 'clause': {'clause_type': 'other', 'risk_domains': ['eye', 'skin']},
 'indexing': {'chunk_id': 'chunk_000860',
              'chunk_char_len': 252,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'simple',
              'tag_method': 'rule',
              'tag_confidence': 0.25}},
)
