from langchain_core.documents import Document

chunk = Document(
    page_content=('rowspan="22">질환</td><td>FBA002</td><td>각막 '
 '이영양증</td></tr><tr><td>FBA003</td><td>기타 각막염 (판누스 '
 '포함)</td></tr><tr><td>FBA004</td><td>각막염(비궤양성)</td></tr><tr><td>FCA001</td><td>건성 '
 '각결막염 · KCS</td></tr><tr><td>FCA002</td><td>결막염 (결막 부종 '
 '포함)</td></tr><tr><td>FDA001</td><td>포도막염 (홍채염 / 전안방 출혈'),
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
 'indexing': {'chunk_id': 'chunk_000856',
              'chunk_char_len': 280,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'simple',
              'tag_method': 'rule',
              'tag_confidence': 0.25}},
)
