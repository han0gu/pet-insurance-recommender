from langchain_core.documents import Document

chunk = Document(
    page_content=('- 2. 계약자, 피보험자 또는 보험수익자의 책임있는 사유에 의하는 경우 : 이미 경과한\n'
 '- 기간에 대하여 <부표3> ‘단기요율표’에서 정한 단기요율(1년 미만의 기간에 적용되\n'
 '- 는 요율)로 계산한 보험료를 뺀 잔액. 다만, 계약자, 피보험자 또는 보험수익자의 고\n'
 '- 의 또는 중대한 과실로 무효가 된 때에는 보험료를 돌려드리지 아니합니다.\n'
 '- ② 보험기간이 1년을 초과하는 계약이 무효, 효력상실 또는 소멸인 경우에는 무효, 효력상\n'
 '- 실 또는 소멸의 원인이 생긴 날 또는 해지일이 속하는 보험연도의 보험료는 제1항의'),
    metadata={'source_doc': {'total_pages': 50},
 'doc': {'doc_type': 'terms',
         'file_name': 'meritz_1_maum_pet_1_21_22_50.pdf',
         'insurer_code': 'meritz',
         'product_code': '1',
         'product_name': '메리츠 마음든든 반려동물보험',
         'total_pages': 1,
         'page': 1},
 'term_type': 'unknown',
 'clause': {'clause_type': 'other', 'risk_domains': ['other']},
 'indexing': {'chunk_id': 'chunk_000107',
              'chunk_char_len': 291,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'normal',
              'tag_method': 'rule',
              'tag_confidence': 0.25}},
)
