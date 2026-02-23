from langchain_core.documents import Document

chunk = Document(
    page_content=('비대성 심근병증</td></tr><tr><td rowspan="8">4</td><td '
 'rowspan="8">비뇨기과</td><td>AGA001</td><td>신장의 양성 '
 '신생물</td></tr><tr><td>AGB001</td><td>신장의 악성 '
 '신생물</td></tr><tr><td>AGC001</td><td>신장의 신생물 (양성 또는 악성이 '
 '불확실한)</td></tr><tr><td>AGB002</td><td>이행상피세포암종 '
 '(방광)</td></tr><tr><td>AGA003</td><td>기타 방광의 양성 신생물'),
    metadata={'source_doc': {'total_pages': 160},
 'doc': {'doc_type': 'terms',
         'file_name': 'meritz_2_petpermint_cat_family_45_82_83_206.pdf',
         'insurer_code': 'meritz',
         'product_code': '2',
         'product_name': '무배당 펫퍼민트 Cat&Family보험 다이렉트2601',
         'total_pages': 1,
         'page': 1},
 'term_type': 'unknown',
 'clause': {'clause_type': 'other', 'risk_domains': ['digestive', 'urinary']},
 'indexing': {'chunk_id': 'chunk_000864',
              'chunk_char_len': 290,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'simple',
              'tag_method': 'rule',
              'tag_confidence': 0.25}},
)
