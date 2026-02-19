from langchain_core.documents import Document

chunk = Document(
    page_content=('※ 주) 가관절이란, 충분한 경과 및 골이식술 등 골 유합을 얻는데 필요한 수술적 치료를 시행하 였음에도 불구하고 골절부의 유합이 '
 '이루어지 지 않는 ‘불유합’ 상태를 말하며, 골유합이 지연되는 지연유합은 제외한다.'),
    metadata={'source_doc': {'total_pages': 160},
 'doc': {'doc_type': 'terms',
         'file_name': 'meritz_2_petpermint_cat_family_45_82_83_206.pdf',
         'insurer_code': 'meritz',
         'product_code': '2',
         'product_name': '무배당 펫퍼민트 Puppy&Family보험 다이렉트2601',
         'total_pages': 160,
         'page': 195},
 'term_type': 'special',
 'clause': {'clause_type': 'exclusion', 'risk_domains': ['joint']},
 'indexing': {'chunk_id': 'chunk_000706',
              'chunk_char_len': 119,
              'embedding_model': 'solar-embedding-1-large',
              'tag_method': 'rule',
              'tag_confidence': 0.85}},
)
