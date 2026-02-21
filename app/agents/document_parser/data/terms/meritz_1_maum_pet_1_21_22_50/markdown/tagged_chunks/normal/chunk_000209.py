from langchain_core.documents import Document

chunk = Document(
    page_content=('한명은 장애인인 경우: 모든 보험수익자가 장애인이 아니므로 이 특별약관을 적용할 수 없습니다.2. 전환대상계약의 보험수익자 1인은 '
 '비장애인이고 피보험자 2인 중 한명은 비장애인,\n'
 '한명은 장애인인 경우- \n'
 ': 모든 피보험자가 장애인이 아니므로 이 특별약관을 적용할 수 없습니다.3. 전환대상계약의 피보험자는 비장애인이고 보험수익자가 '
 '법정상속인(장애인)인 경우: 현재 법정상속인이 장애인이라고 하더라도 이 특별약관을 적용할 수 없습니다. 장'),
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
 'indexing': {'chunk_id': 'chunk_000209',
              'chunk_char_len': 244,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'normal',
              'tag_method': 'rule',
              'tag_confidence': 0.25}},
)
