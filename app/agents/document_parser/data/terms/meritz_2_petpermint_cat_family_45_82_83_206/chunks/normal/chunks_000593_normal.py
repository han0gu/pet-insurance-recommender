from langchain_core.documents import Document

chunk = Document(
    page_content=('구 분 | 특정질병 | 분류코드 | 항목명\n'
 '질환 | FBA002 | 각막 이영양증\n'
 'FBA003 | 기타 각막염 (판누스 포함)\n'
 'FBA004 | 각막염(비궤양성)\n'
 'FCA001 | 건성 각결막염 · KCS\n'
 'FCA002 | 결막염 (결막 부종 포함)\n'
 'FDA001 | 포도막염 (홍채염 / 전안방 출혈 포함)\n'
 'FEA001 | 백내장 (좌안)\n'
 'FEA002 | 백내장 (우안)\n'
 'FEA003 FEA004 | 수정체 (아) 탈구 백내장\n'
 'FFA001 | 망막 변성 / 망막 위축 / PRA\n'
 'FFA002 | 망막 박리 (유리체 변성 포함)'),
    metadata={'source_doc': {'total_pages': 160},
 'doc': {'doc_type': 'terms',
         'file_name': 'meritz_2_petpermint_cat_family_45_82_83_206.pdf',
         'insurer_code': 'meritz',
         'product_code': '2',
         'product_name': '무배당 펫퍼민트 Puppy&Family보험 다이렉트2601',
         'total_pages': 160,
         'page': 170},
 'term_type': 'special',
 'clause': {'clause_type': 'definition', 'risk_domains': ['eye']},
 'indexing': {'chunk_id': 'chunk_000593',
              'chunk_char_len': 291,
              'embedding_model': 'solar-embedding-1-large',
              'tag_method': 'llm',
              'tag_confidence': 0.95}},
)
