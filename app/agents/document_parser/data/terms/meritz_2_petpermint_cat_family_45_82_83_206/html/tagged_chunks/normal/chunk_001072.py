from langchain_core.documents import Document

chunk = Document(
    page_content=('장 해를 남긴 때</td><td>50</td></tr><tr><td>4) 흉복부장기 또는 비뇨생식기 기능에 뚜렷한 장해를 남긴 '
 '때</td><td>30</td></tr><tr><td>5) 흉복부장기 또는 비뇨생식기 기능에 약간의 장해를 남긴 '
 "때</td><td>15</td></tr></tbody></table><h1 id='16' style='font-size:20px'>나"),
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
 'indexing': {'chunk_id': 'chunk_001072',
              'chunk_char_len': 208,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'normal',
              'tag_method': 'rule',
              'tag_confidence': 0.25}},
)
