from langchain_core.documents import Document

chunk = Document(
    page_content='. 4) 코의 추상(추한 모습)장해를 수반한 때에는 기능장해의 지급률과 추상(추한 모습)장해의 지급률을 합산한다.',
    metadata={'source_doc': {'total_pages': 180},
 'doc': {'doc_type': 'terms',
         'file_name': 'meritz_3_petpermint_dog_family_51_86_87_230.pdf',
         'insurer_code': 'meritz',
         'product_code': '3',
         'product_name': '무배당 펫퍼민트 Puppy&Family보험 다이렉트2601',
         'total_pages': 180,
         'page': 206},
 'term_type': 'special',
 'clause': {'clause_type': 'coverage', 'risk_domains': ['head']},
 'indexing': {'chunk_id': 'chunk_000721',
              'chunk_char_len': 63,
              'embedding_model': 'solar-embedding-1-large',
              'tag_method': 'llm',
              'tag_confidence': 0.95}},
)
