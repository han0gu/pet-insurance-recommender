from langchain_core.documents import Document

chunk = Document(
    page_content=('12회이상)</td><td>6</td></tr><tr><td>장기 통원치료(1년간 '
 '6회이상)</td><td>4</td></tr><tr><td>단기 통원치료(6개월간 '
 '6회이상)</td><td>2</td></tr><tr><td>단기 통원치료(6개월간 '
 '6회미만)</td><td>0</td></tr><tr><td rowspan="3">기능 장해 소견</td><td>두 눈을 감고 일어서기 '
 '곤란하거나 두 눈 을 뜨고 10m 거리를 직선으로 걷다가'),
    metadata={'source_doc': {'total_pages': 160},
 'doc': {'doc_type': 'terms',
         'file_name': 'meritz_2_petpermint_cat_family_45_82_83_206.pdf',
         'insurer_code': 'meritz',
         'product_code': '2',
         'product_name': '무배당 펫퍼민트 Cat&Family보험 다이렉트2601',
         'total_pages': 1,
         'page': 1},
 'term_type': 'unknown',
 'clause': {'clause_type': 'other', 'risk_domains': ['digestive', 'eye']},
 'indexing': {'chunk_id': 'chunk_000940',
              'chunk_char_len': 244,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'normal',
              'tag_method': 'rule',
              'tag_confidence': 0.25}},
)
