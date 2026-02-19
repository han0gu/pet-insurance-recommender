from langchain_core.documents import Document

chunk = Document(
    page_content=('. 4) “한눈의 교정시력이 0.02이하로 된 때”라 함은 안 전수동(Hand Movement)주1), 안전수지(Finger '
 'Counting)주2) 상태를 포함한다.'),
    metadata={'source_doc': {'total_pages': 180},
 'doc': {'doc_type': 'terms',
         'file_name': 'meritz_3_petpermint_dog_family_51_86_87_230.pdf',
         'insurer_code': 'meritz',
         'product_code': '3',
         'product_name': '무배당 펫퍼민트 Puppy&Family보험 다이렉트2601',
         'total_pages': 180,
         'page': 203},
 'term_type': 'special',
 'clause': {'clause_type': 'exclusion', 'risk_domains': ['eye']},
 'indexing': {'chunk_id': 'chunk_000705',
              'chunk_char_len': 92,
              'embedding_model': 'solar-embedding-1-large',
              'tag_method': 'llm',
              'tag_confidence': 0.9}},
)
