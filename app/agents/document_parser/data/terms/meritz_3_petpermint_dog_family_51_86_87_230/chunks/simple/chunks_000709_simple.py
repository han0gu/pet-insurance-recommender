from langchain_core.documents import Document

chunk = Document(
    page_content=('. 9) “눈꺼풀에 뚜렷한 결손을 남긴 때”라 함은 눈꺼풀 의 결손으로 눈을 감았을 때 각막(검은 자위)이 완전 히 덮이지 않는 경우를 '
 '말한다.'),
    metadata={'source_doc': {'total_pages': 180},
 'doc': {'doc_type': 'terms',
         'file_name': 'meritz_3_petpermint_dog_family_51_86_87_230.pdf',
         'insurer_code': 'meritz',
         'product_code': '3',
         'product_name': '무배당 펫퍼민트 Puppy&Family보험 다이렉트2601',
         'total_pages': 180,
         'page': 203},
 'term_type': 'special',
 'clause': {'clause_type': 'coverage', 'risk_domains': ['eye', 'skin']},
 'indexing': {'chunk_id': 'chunk_000709',
              'chunk_char_len': 80,
              'embedding_model': 'solar-embedding-1-large',
              'tag_method': 'llm',
              'tag_confidence': 0.95}},
)
