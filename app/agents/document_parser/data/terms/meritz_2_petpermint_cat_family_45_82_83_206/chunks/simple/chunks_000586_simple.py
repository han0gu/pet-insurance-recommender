from langchain_core.documents import Document

chunk = Document(
    page_content=('는 보험계약의 부활(효력회복)을 승낙한 경우에 한하여 보 통약관 제30조(보험료의 납입을 연체하여 해지된 계약의 부 활(효력회복))를 '
 '준용합니다.\n'
 '제4조(준용규정)\n'
 '이 특별약관에 정하지 않은 사항은 보통약관 및 해당 특별 약관을 따릅니다.'),
    metadata={'source_doc': {'total_pages': 160},
 'doc': {'doc_type': 'terms',
         'file_name': 'meritz_2_petpermint_cat_family_45_82_83_206.pdf',
         'insurer_code': 'meritz',
         'product_code': '2',
         'product_name': '무배당 펫퍼민트 Puppy&Family보험 다이렉트2601',
         'total_pages': 160,
         'page': 168},
 'term_type': 'special',
 'clause': {'clause_type': 'other', 'risk_domains': ['other']},
 'indexing': {'chunk_id': 'chunk_000586',
              'chunk_char_len': 133,
              'embedding_model': 'solar-embedding-1-large',
              'tag_method': 'llm',
              'tag_confidence': 0.95}},
)
