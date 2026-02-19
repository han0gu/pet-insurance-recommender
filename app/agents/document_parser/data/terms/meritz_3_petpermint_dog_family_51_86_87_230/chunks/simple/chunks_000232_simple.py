from langchain_core.documents import Document

chunk = Document(
    page_content=('\uf000 반려동물 비용손해 관련 특별약관을 체결할 때 이 특별 약관에서 정한 피보험자 및 반려동물의 나이에 미달되었거 나 초과되었을 '
 '경우에는 계약을 무효로 하며 이미 납입한 보험료를 돌려드립니다. 다만, 회사가 나이의 착오를 발견 하였을 때 이미 계약나이에 도달한 '
 '경우에는 유효한 계약으 로 봅니다'),
    metadata={'source_doc': {'total_pages': 180},
 'doc': {'doc_type': 'terms',
         'file_name': 'meritz_3_petpermint_dog_family_51_86_87_230.pdf',
         'insurer_code': 'meritz',
         'product_code': '3',
         'product_name': '무배당 펫퍼민트 Puppy&Family보험 다이렉트2601',
         'total_pages': 180,
         'page': 100},
 'term_type': 'special',
 'clause': {'clause_type': 'other', 'risk_domains': ['other']},
 'indexing': {'chunk_id': 'chunk_000232',
              'chunk_char_len': 163,
              'embedding_model': 'solar-embedding-1-large',
              'tag_method': 'llm',
              'tag_confidence': 0.95}},
)
