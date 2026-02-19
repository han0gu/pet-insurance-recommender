from langchain_core.documents import Document

chunk = Document(
    page_content=('제22조(준용규정)\n'
 '이「반려동물 비용손해 관련 특별약관 일반조항」에서 정하 지 않은 사항은 보통약관을 따릅니다. 다만, 보통약관 제3 조(보험금의 '
 '지급사유), 제4조(보험금 지급에 관한 세부규 정), 제5조(보험금을 지급하지 않는 사유), 제9조(적립부분 적립이율에 관한 사항), '
 '제10조(만기환급금의 지급), 제25'),
    metadata={'source_doc': {'total_pages': 160},
 'doc': {'doc_type': 'terms',
         'file_name': 'meritz_2_petpermint_cat_family_45_82_83_206.pdf',
         'insurer_code': 'meritz',
         'product_code': '2',
         'product_name': '무배당 펫퍼민트 Puppy&Family보험 다이렉트2601',
         'total_pages': 160,
         'page': 103},
 'term_type': 'special',
 'clause': {'clause_type': 'exclusion', 'risk_domains': ['other']},
 'indexing': {'chunk_id': 'chunk_000281',
              'chunk_char_len': 177,
              'embedding_model': 'solar-embedding-1-large',
              'tag_method': 'rule',
              'tag_confidence': 0.85}},
)
