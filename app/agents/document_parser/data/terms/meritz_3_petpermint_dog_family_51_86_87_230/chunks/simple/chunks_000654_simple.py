from langchain_core.documents import Document

chunk = Document(
    page_content=('제5조(자동갱신 적용대상 계약의 보장개시)\n'
 '제2조(자동갱신 적용대상 계약의 자동갱신)에 따라 계약이 갱신되는 경우 갱신보장계약의 보장개시는 갱신일 당일부터 개시됩니다.'),
    metadata={'source_doc': {'total_pages': 180},
 'doc': {'doc_type': 'terms',
         'file_name': 'meritz_3_petpermint_dog_family_51_86_87_230.pdf',
         'insurer_code': 'meritz',
         'product_code': '3',
         'product_name': '무배당 펫퍼민트 Puppy&Family보험 다이렉트2601',
         'total_pages': 180,
         'page': 190},
 'term_type': 'special',
 'clause': {'clause_type': 'coverage', 'risk_domains': ['digestive']},
 'indexing': {'chunk_id': 'chunk_000654',
              'chunk_char_len': 93,
              'embedding_model': 'solar-embedding-1-large',
              'tag_method': 'rule',
              'tag_confidence': 0.8}},
)
