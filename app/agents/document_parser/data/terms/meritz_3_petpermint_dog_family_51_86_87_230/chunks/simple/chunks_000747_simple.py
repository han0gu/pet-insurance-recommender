from langchain_core.documents import Document

chunk = Document(
    page_content='. 6) 심한 운동장해란 다음 중 어느 하나에 해당하는 경우를 말한다.',
    metadata={'source_doc': {'total_pages': 180},
 'doc': {'doc_type': 'terms',
         'file_name': 'meritz_3_petpermint_dog_family_51_86_87_230.pdf',
         'insurer_code': 'meritz',
         'product_code': '3',
         'product_name': '무배당 펫퍼민트 Puppy&Family보험 다이렉트2601',
         'total_pages': 180,
         'page': 211},
 'term_type': 'special',
 'clause': {'clause_type': 'other', 'risk_domains': ['joint', 'other']},
 'indexing': {'chunk_id': 'chunk_000747',
              'chunk_char_len': 39,
              'embedding_model': 'solar-embedding-1-large',
              'tag_method': 'llm',
              'tag_confidence': 0.9}},
)
