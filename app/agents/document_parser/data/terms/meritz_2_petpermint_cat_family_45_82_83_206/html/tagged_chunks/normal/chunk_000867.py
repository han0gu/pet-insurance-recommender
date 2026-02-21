from langchain_core.documents import Document

chunk = Document(
    page_content=('비뇨기계 신생물 (양성 또는 악성이 불 확실한)</td></tr><tr><td>OAA001</td><td>급성 '
 '신부전</td></tr><tr><td>OAA002</td><td>신우 '
 '신염</td></tr><tr><td>OAA003</td><td>수신증</td></tr><tr><td>OAA004</td><td>만성 신장 '
 '질환 (신부전 포함)</td></tr><tr><td>OAA005</td><td>신장 결석</td></tr><tr><td>OAA006 '
 'OAA007</td><td>방광염 방광'),
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
 'indexing': {'chunk_id': 'chunk_000867',
              'chunk_char_len': 273,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'normal',
              'tag_method': 'rule',
              'tag_confidence': 0.25}},
)
