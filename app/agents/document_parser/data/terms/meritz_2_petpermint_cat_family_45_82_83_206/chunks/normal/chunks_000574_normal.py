from langchain_core.documents import Document

chunk = Document(
    page_content=('료를 지급합니다.\n'
 '제7조(준용규정)\n'
 '이 특별약관에서 정하지 않은 사항은「반려동물 비용손해 관련 특별약관 일반조항」을 따르고,「반려동물 비용손해 관련 특별약관 일반조항」에서 '
 '정하지 않은 사항은 보통약 관을 따릅니다.'),
    metadata={'source_doc': {'total_pages': 160},
 'doc': {'doc_type': 'terms',
         'file_name': 'meritz_2_petpermint_cat_family_45_82_83_206.pdf',
         'insurer_code': 'meritz',
         'product_code': '2',
         'product_name': '무배당 펫퍼민트 Puppy&Family보험 다이렉트2601',
         'total_pages': 160,
         'page': 165},
 'term_type': 'special',
 'clause': {'clause_type': 'other', 'risk_domains': ['other']},
 'indexing': {'chunk_id': 'chunk_000574',
              'chunk_char_len': 119,
              'embedding_model': 'solar-embedding-1-large',
              'tag_method': 'llm',
              'tag_confidence': 0.9}},
)
