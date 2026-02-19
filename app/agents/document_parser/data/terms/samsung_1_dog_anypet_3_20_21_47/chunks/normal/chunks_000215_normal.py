from langchain_core.documents import Document

chunk = Document(
    page_content=('<소득세법 시행규칙 제54조(장애아동의 범위) >\n'
 '영 제107조제1항제1호에서 "기획재정부령으로 정하는 사람"이란 「장애아동 복지지원법」 제21 조제1 항에 따른 발달재활서비스를 지원받고 '
 '있는 사람을 말한다.\n'
 '【예시】\n'
 '<이 특별약관을 적용할 수 없는 사례 예시 1>\n'
 '전환대상계약의 피보험자 1인은 비장애인이고 보험수익자 2인 중 한명은 비장애인, 한명은 장애인인 경우 ⇒ 모든 보험수익자가 장애인이 '
 '아니므로 이 특별약관을 적용할 수 없습니다.\n'
 '<이 특별약관을 적용할 수 없는 사례 예시 2>'),
    metadata={'source_doc': {'total_pages': 45},
 'doc': {'doc_type': 'terms',
         'file_name': 'samsung_1_dog_anypet_3_20_21_47.pdf',
         'insurer_code': 'samsung',
         'product_code': '1',
         'product_name': '(일반)반려견보험 애니펫',
         'total_pages': 45,
         'page': 44},
 'term_type': 'special',
 'clause': {'clause_type': 'other', 'risk_domains': ['other']},
 'indexing': {'chunk_id': 'chunk_000215',
              'chunk_char_len': 274,
              'embedding_model': 'solar-embedding-1-large',
              'tag_method': 'llm',
              'tag_confidence': 0.95}},
)
