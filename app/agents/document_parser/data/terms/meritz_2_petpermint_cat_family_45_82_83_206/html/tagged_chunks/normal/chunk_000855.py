from langchain_core.documents import Document

chunk = Document(
    page_content=("각막염 · 각막궤양 (각막 미란 포함)</td></tr></tbody></table><footer id='13' "
 "style='font-size:14px'>169</footer><table id='14' "
 "style='font-size:14px'><thead><tr><td>구 "
 '분</td><td>특정질병</td><td>분류코드</td><td>항목명</td></tr></thead><tbody><tr><td '
 'rowspan="22"></td><td rowspan="22">질환</td><td>FBA002</td><td>각막'),
    metadata={'source_doc': {'total_pages': 160},
 'doc': {'doc_type': 'terms',
         'file_name': 'meritz_2_petpermint_cat_family_45_82_83_206.pdf',
         'insurer_code': 'meritz',
         'product_code': '2',
         'product_name': '무배당 펫퍼민트 Cat&Family보험 다이렉트2601',
         'total_pages': 1,
         'page': 1},
 'term_type': 'unknown',
 'clause': {'clause_type': 'other', 'risk_domains': ['eye']},
 'indexing': {'chunk_id': 'chunk_000855',
              'chunk_char_len': 288,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'normal',
              'tag_method': 'rule',
              'tag_confidence': 0.25}},
)
