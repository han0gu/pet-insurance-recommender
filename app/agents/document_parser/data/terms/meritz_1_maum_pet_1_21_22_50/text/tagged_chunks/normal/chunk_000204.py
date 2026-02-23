from langchain_core.documents import Document

chunk = Document(
    page_content=('한명은 장애인인 경우: 모든 피보험자가 장애인이 아니므로 이 특별약관을 적용할 수 없습니다.3. 전환대상계약의 피보험자는 비장애인이고 '
 '보험수익자가 법정상속인(장애인)인 경우: 현재 법정상속인이 장애인이라고 하더라도 이 특별약관을 적용할 수 없습니다. 장\n'
 '애인전용보험으로 전환을 원할 경우 수익자 지정이 필요합니다.② 전환대상계약이 해지(解止) 또는 기타 사유로 효력이 없게 된 경우 또는 '
 '전환대상계약\n'
 '이 제1항에서 정한 조건을 만족하지 않게 된 경우 이 특별약관은 그 때부터 효력이 없\n'
 '습니다.'),
    metadata={'source_doc': {'total_pages': 50},
 'doc': {'doc_type': 'terms',
         'file_name': 'meritz_1_maum_pet_1_21_22_50.pdf',
         'insurer_code': 'meritz',
         'product_code': '1',
         'product_name': '메리츠 마음든든 반려동물보험',
         'total_pages': 1,
         'page': 1},
 'term_type': 'unknown',
 'clause': {'clause_type': 'other', 'risk_domains': ['digestive']},
 'indexing': {'chunk_id': 'chunk_000204',
              'chunk_char_len': 276,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'normal',
              'tag_method': 'rule',
              'tag_confidence': 0.25}},
)
