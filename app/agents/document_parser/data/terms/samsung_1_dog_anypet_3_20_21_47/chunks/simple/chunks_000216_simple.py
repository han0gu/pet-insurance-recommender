from langchain_core.documents import Document

chunk = Document(
    page_content=('<이 특별약관을 적용할 수 없는 사례 예시 2>\n'
 '전환대상계약의 보험수익자 1인은 비장애인이고 피보험자 2인 중 한명은 비장애인, 한명은 장애인인 경우 ⇒ 모든 피보험자가 장애인이 '
 '아니므로 이 특별약관을 적용할 수 없습니다.\n'
 '<이 특별약관을 적용할 수 없는 사례 예시 3>\n'
 '전환대상계약의 피보험자는 비장애인이고 보험수익자가 법정상속인(장애인)인 경우 ⇒ 현재 법정상속인이 장애인이라고 하더라도 이 특별약관을 '
 '적용할 수 없습니다. 장애인전용보험으로 전환을 원할 경우 수익자 지정이 필요합니다.'),
    metadata={'source_doc': {'total_pages': 45},
 'doc': {'doc_type': 'terms',
         'file_name': 'samsung_1_dog_anypet_3_20_21_47.pdf',
         'insurer_code': 'samsung',
         'product_code': '1',
         'product_name': '(일반)반려견보험 애니펫',
         'total_pages': 45,
         'page': 44},
 'term_type': 'special',
 'clause': {'clause_type': 'exclusion', 'risk_domains': ['other']},
 'indexing': {'chunk_id': 'chunk_000216',
              'chunk_char_len': 273,
              'embedding_model': 'solar-embedding-1-large',
              'tag_method': 'llm',
              'tag_confidence': 0.95}},
)
