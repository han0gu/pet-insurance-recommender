from langchain_core.documents import Document

chunk = Document(
    page_content=('는 회사에 납입한 보험료의 전액, 효력상실, 해지 또는 소멸의 경우에는 경과하지\n'
 '아니한 기간에 대하여 일단위로 계산한 보험료\n'
 '2. 계약자, 피보험자 또는 보험수익자의 책임있는 사유에 의하는 경우 : 이미 경과한\n'
 '기간에 대하여 <부표3> ‘단기요율표’에서 정한 단기요율(1년 미만의 기간에 적용되\n'
 '는 요율)로 계산한 보험료를 뺀 잔액. 다만, 계약자, 피보험자 또는 보험수익자의 고'),
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
 'indexing': {'chunk_id': 'chunk_000102',
              'chunk_char_len': 212,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'simple',
              'tag_method': 'rule',
              'tag_confidence': 0.25}},
)
