from langchain_core.documents import Document

chunk = Document(
    page_content=('비뇨기질환 확장보장 특별약관\n'
 '제1 조(보상하는 손해)\n'
 '회사는 보통약관 제5조(보상하지 않는 손해) 제2항 제1호에도 불구하고, 비뇨기질환(요로결석 등) 을 원인으로 하여 생긴 반려동물의 '
 '치료비를 보통약관 제4조(보상하는 손해)에 따라 보상하여 드 립니다.\n'
 '제2조(준용규정)\n'
 '이 특별약관에 정하지 않은 사항은 보통약관을 따릅니다.'),
    metadata={'source_doc': {'total_pages': 35},
 'doc': {'doc_type': 'terms',
         'file_name': 'samsung_2_cat_anypet_3_20_21_37.pdf',
         'insurer_code': 'samsung',
         'product_code': '2',
         'product_name': '(일반)반려묘보험 애니펫',
         'total_pages': 35,
         'page': 22},
 'term_type': 'special',
 'clause': {'clause_type': 'exclusion',
            'risk_domains': ['digestive', 'urinary']},
 'indexing': {'chunk_id': 'chunk_000114',
              'chunk_char_len': 183,
              'embedding_model': 'solar-embedding-1-large',
              'tag_method': 'rule',
              'tag_confidence': 0.85}},
)
