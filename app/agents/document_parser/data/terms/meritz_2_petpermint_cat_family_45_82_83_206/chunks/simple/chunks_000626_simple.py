from langchain_core.documents import Document

chunk = Document(
    page_content=('8) 한 눈에 뚜렷한 시야장해를 남긴 때 | 5\n'
 '9) 한눈의 눈꺼풀에 뚜렷한 결손을 남긴 때 | 10\n'
 '10) 한눈의 눈꺼풀에 뚜렷한 운동장해를 남긴 때 | 5'),
    metadata={'source_doc': {'total_pages': 160},
 'doc': {'doc_type': 'terms',
         'file_name': 'meritz_2_petpermint_cat_family_45_82_83_206.pdf',
         'insurer_code': 'meritz',
         'product_code': '2',
         'product_name': '무배당 펫퍼민트 Puppy&Family보험 다이렉트2601',
         'total_pages': 160,
         'page': 177},
 'term_type': 'special',
 'clause': {'clause_type': 'coverage', 'risk_domains': ['eye']},
 'indexing': {'chunk_id': 'chunk_000626',
              'chunk_char_len': 88,
              'embedding_model': 'solar-embedding-1-large',
              'tag_method': 'llm',
              'tag_confidence': 0.95}},
)
