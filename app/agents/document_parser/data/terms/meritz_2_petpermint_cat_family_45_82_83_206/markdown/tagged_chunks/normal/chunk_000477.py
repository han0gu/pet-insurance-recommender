from langchain_core.documents import Document

chunk = Document(
    page_content=('|  | 질환 | FBA002 | 각막 이영양증 |\n'
 '| FBA003 | 질환 | 기타 각막염 (판누스 포함) |  |\n'
 '| FBA004 | 질환 | 각막염(비궤양성) |  |\n'
 '| FCA001 | 질환 | 건성 각결막염 · KCS |  |\n'
 '| FCA002 | 질환 | 결막염 (결막 부종 포함) |  |\n'
 '| FDA001 | 질환 | 포도막염 (홍채염 / 전안방 출혈 포함) |  |\n'
 '| FEA001 | 질환 | 백내장 (좌안) |  |\n'
 '| FEA002 | 질환 | 백내장 (우안) |  |'),
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
 'indexing': {'chunk_id': 'chunk_000477',
              'chunk_char_len': 271,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'normal',
              'tag_method': 'rule',
              'tag_confidence': 0.25}},
)
