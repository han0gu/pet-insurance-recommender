from langchain_core.documents import Document

chunk = Document(
    page_content=('제5조(준용규정)\n'
 '이 특별약관에서 정하지 않은 사항은「배상책임 관련 특별 약관 일반조항」을 따르고,「배상책임 관련 특별약관 일반 조항」에서 정하지 않은 '
 '사항은「반려동물 비용손해 관련 특별약관 일반조항」을 따릅니다. 단,「반려동물 비용손해 관련 특별약관 일반조항」에서 정하지 않은 사항은 '
 '보통약 관을 따릅니다.'),
    metadata={'source_doc': {'total_pages': 180},
 'doc': {'doc_type': 'terms',
         'file_name': 'meritz_3_petpermint_dog_family_51_86_87_230.pdf',
         'insurer_code': 'meritz',
         'product_code': '3',
         'product_name': '무배당 펫퍼민트 Puppy&Family보험 다이렉트2601',
         'total_pages': 180,
         'page': 188},
 'term_type': 'special',
 'clause': {'clause_type': 'other', 'risk_domains': ['other']},
 'indexing': {'chunk_id': 'chunk_000643',
              'chunk_char_len': 172,
              'embedding_model': 'solar-embedding-1-large',
              'tag_method': 'llm',
              'tag_confidence': 0.95}},
)
