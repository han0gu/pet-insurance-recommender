from langchain_core.documents import Document

chunk = Document(
    page_content=('장해를 남긴 때</td><td>60</td></tr><tr><td>4) 씹어먹는 기능과 말하는 기능 모두에 뚜렷한 장해를 남긴 '
 '때</td><td>40</td></tr><tr><td>5) 씹어먹는 기능 또는 말하는 기능에 뚜렷한 장 해를 남긴 '
 '때</td><td>20</td></tr><tr><td>6) 씹어먹는 기능과 말하는 기능 모두에 약간의 장해를 남긴 '
 '때</td><td>10</td></tr><tr><td>7) 씹어먹는 기능 또는 말하는 기능에 약간의 장 해를 남긴 '
 '때</td><td>5</td></tr><tr><td>8) 치아에'),
    metadata={'source_doc': {'total_pages': 160},
 'doc': {'doc_type': 'terms',
         'file_name': 'meritz_2_petpermint_cat_family_45_82_83_206.pdf',
         'insurer_code': 'meritz',
         'product_code': '2',
         'product_name': '무배당 펫퍼민트 Cat&Family보험 다이렉트2601',
         'total_pages': 1,
         'page': 1},
 'term_type': 'unknown',
 'clause': {'clause_type': 'other', 'risk_domains': ['dental', 'digestive']},
 'indexing': {'chunk_id': 'chunk_000950',
              'chunk_char_len': 298,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'simple',
              'tag_method': 'rule',
              'tag_confidence': 0.25}},
)
