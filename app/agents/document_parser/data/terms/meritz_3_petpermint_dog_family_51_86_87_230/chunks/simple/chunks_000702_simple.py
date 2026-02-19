from langchain_core.documents import Document

chunk = Document(
    page_content=('8) 한 눈에 뚜렷한 시야장해를 남긴 때 | 5\n'
 '9) 한눈의 눈꺼풀에 뚜렷한 결손을 남긴 때 | 10\n'
 '10) 한눈의 눈꺼풀에 뚜렷한 운동장해를 남긴 때 | 5'),
    metadata={'source_doc': {'total_pages': 180},
 'doc': {'doc_type': 'terms',
         'file_name': 'meritz_3_petpermint_dog_family_51_86_87_230.pdf',
         'insurer_code': 'meritz',
         'product_code': '3',
         'product_name': '무배당 펫퍼민트 Puppy&Family보험 다이렉트2601',
         'total_pages': 180,
         'page': 202},
 'term_type': 'special',
 'clause': {'clause_type': 'coverage', 'risk_domains': ['eye']},
 'indexing': {'chunk_id': 'chunk_000702',
              'chunk_char_len': 88,
              'embedding_model': 'solar-embedding-1-large',
              'tag_method': 'llm',
              'tag_confidence': 0.95}},
)
