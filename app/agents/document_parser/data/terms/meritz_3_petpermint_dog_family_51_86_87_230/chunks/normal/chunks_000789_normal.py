from langchain_core.documents import Document

chunk = Document(
    page_content=('. 6) “손가락뼈 일부를 잃었을 때”라 함은 첫째 손가락 의 지관절, 다른 네 손가락의 제1지관절(근위지관 절)부터 심장에서 먼쪽으로 '
 '손가락 뼈의 일부가 절 단된 경우를 말하며, 뼈 단면이 불규칙해진 상태나 손가락 길이의 단축 없이 골편만 떨어진 상태는 해당 하지 않는다'),
    metadata={'source_doc': {'total_pages': 180},
 'doc': {'doc_type': 'terms',
         'file_name': 'meritz_3_petpermint_dog_family_51_86_87_230.pdf',
         'insurer_code': 'meritz',
         'product_code': '3',
         'product_name': '무배당 펫퍼민트 Puppy&Family보험 다이렉트2601',
         'total_pages': 180,
         'page': 221},
 'term_type': 'special',
 'clause': {'clause_type': 'definition', 'risk_domains': ['joint']},
 'indexing': {'chunk_id': 'chunk_000789',
              'chunk_char_len': 152,
              'embedding_model': 'solar-embedding-1-large',
              'tag_method': 'llm',
              'tag_confidence': 0.95}},
)
