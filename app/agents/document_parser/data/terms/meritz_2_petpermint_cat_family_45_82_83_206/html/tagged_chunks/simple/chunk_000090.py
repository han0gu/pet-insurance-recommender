from langchain_core.documents import Document

chunk = Document(
    page_content=('위험변경사항 통지\n'
 '(우편, 전화, 방문 등)\n'
 '↓\n'
 '계약자,피보험자의 계약변경사항 확인 후 청약\n'
 '↓\n'
 '계약변경사항 인수 심사\n'
 '↓\n'
 '정산금액 처리\n'
 '(환급 또는 추가납입)\n'
 '↓'),
    metadata={'source_doc': {'total_pages': 160},
 'doc': {'doc_type': 'terms',
         'file_name': 'meritz_2_petpermint_cat_family_45_82_83_206.pdf',
         'insurer_code': 'meritz',
         'product_code': '2',
         'product_name': '무배당 펫퍼민트 Cat&Family보험 다이렉트2601',
         'total_pages': 1,
         'page': 1},
 'term_type': 'unknown',
 'clause': {'clause_type': 'other', 'risk_domains': ['digestive']},
 'indexing': {'chunk_id': 'chunk_000090',
              'chunk_char_len': 91,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'simple',
              'tag_method': 'rule',
              'tag_confidence': 0.25}},
)
