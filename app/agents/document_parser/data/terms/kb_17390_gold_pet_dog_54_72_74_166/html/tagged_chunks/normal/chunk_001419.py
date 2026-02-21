from langchain_core.documents import Document

chunk = Document(
    page_content=('이 특별약관을 적용할 수 없는 사례 ∙ 전환대상계약의 피보험자 1인은 비장애인이고 보험수익자 2인 중 한명은 특 비장애인, 한명은 '
 '장애인인 경우 별 ⇒ 모든 보험수익자가 장애인이 아니므로 이 특별약관을 적용할 수 없습니다'),
    metadata={'source_doc': {'total_pages': 113},
 'doc': {'doc_type': 'terms',
         'file_name': 'kb_17390_gold_pet_dog_54_72_74_166.pdf',
         'insurer_code': 'kb',
         'product_code': '17390',
         'product_name': '[일반보험] KB반려행복펫보험',
         'total_pages': 1,
         'page': 1},
 'term_type': 'unknown',
 'clause': {'clause_type': 'other', 'risk_domains': ['digestive']},
 'indexing': {'chunk_id': 'chunk_001419',
              'chunk_char_len': 123,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'normal',
              'tag_method': 'rule',
              'tag_confidence': 0.25}},
)
