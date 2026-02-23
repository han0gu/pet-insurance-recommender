from langchain_core.documents import Document

chunk = Document(
    page_content=('피부 질환</td></tr><tr><td>QCA001</td><td>귀 가려움증 (원인 '
 '불명)</td></tr><tr><td>QFA001</td><td>발진 (원인 '
 '불명)</td></tr><tr><td>QFA002</td><td>피부염 (원인 '
 '불명)</td></tr><tr><td>QFA003</td><td>피부의 가려움증 (원인 '
 '불명)</td></tr><tr><td>QFA004</td><td>탈모 (원인 불명)</td></tr><tr><td '
 'rowspan="21">6</td><td rowspan="21">소화기'),
    metadata={'source_doc': {'total_pages': 160},
 'doc': {'doc_type': 'terms',
         'file_name': 'meritz_2_petpermint_cat_family_45_82_83_206.pdf',
         'insurer_code': 'meritz',
         'product_code': '2',
         'product_name': '무배당 펫퍼민트 Cat&Family보험 다이렉트2601',
         'total_pages': 1,
         'page': 1},
 'term_type': 'unknown',
 'clause': {'clause_type': 'other', 'risk_domains': ['digestive', 'skin']},
 'indexing': {'chunk_id': 'chunk_000881',
              'chunk_char_len': 288,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'normal',
              'tag_method': 'rule',
              'tag_confidence': 0.25}},
)
