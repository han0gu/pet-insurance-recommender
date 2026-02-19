from langchain_core.documents import Document

chunk = Document(
    page_content=('체(척추뼈 몸통)의 압박골절로 각 척추체(척추뼈 몸통)의 압박률의 합이 60% 이상일 때\n'
 '11) 약간의 기형이란 다음 중 어느 하나에 해당하는 경 우를 말한다.'),
    metadata={'source_doc': {'total_pages': 180},
 'doc': {'doc_type': 'terms',
         'file_name': 'meritz_3_petpermint_dog_family_51_86_87_230.pdf',
         'insurer_code': 'meritz',
         'product_code': '3',
         'product_name': '무배당 펫퍼민트 Puppy&Family보험 다이렉트2601',
         'total_pages': 180,
         'page': 213},
 'term_type': 'special',
 'clause': {'clause_type': 'definition', 'risk_domains': ['joint', 'other']},
 'indexing': {'chunk_id': 'chunk_000755',
              'chunk_char_len': 89,
              'embedding_model': 'solar-embedding-1-large',
              'tag_method': 'llm',
              'tag_confidence': 0.95}},
)
