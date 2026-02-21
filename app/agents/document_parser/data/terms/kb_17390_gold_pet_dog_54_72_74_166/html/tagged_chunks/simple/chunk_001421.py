from langchain_core.documents import Document

chunk = Document(
    page_content=('. ∙ 전환대상계약의 피보험자는 비장애인이고 보험수익자가 법정상속인(장애</td></tr></tbody></table> 인)인 경우 상 '
 '⇒ 현재 법정상속인이 장애인이라고 하더라도 이 특별약관을 적용할 수 없습 해</td></tr></tbody></table><br><p '
 "id='87' data-category='list' style='font-size:14px'>니다"),
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
 'indexing': {'chunk_id': 'chunk_001421',
              'chunk_char_len': 205,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'simple',
              'tag_method': 'rule',
              'tag_confidence': 0.25}},
)
