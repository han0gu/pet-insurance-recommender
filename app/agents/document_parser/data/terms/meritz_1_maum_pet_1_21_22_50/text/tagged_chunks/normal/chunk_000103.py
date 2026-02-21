from langchain_core.documents import Document

chunk = Document(
    page_content=('는 요율)로 계산한 보험료를 뺀 잔액. 다만, 계약자, 피보험자 또는 보험수익자의 고\n'
 '의 또는 중대한 과실로 무효가 된 때에는 보험료를 돌려드리지 아니합니다.② 보험기간이 1년을 초과하는 계약이 무효, 효력상실 또는 소멸인 '
 '경우에는 무효, 효력상\n'
 '실 또는 소멸의 원인이 생긴 날 또는 해지일이 속하는 보험연도의 보험료는 제1항의\n'
 '규정을 적용하고 그 이후의 보험연도에 속하는 보험료는 전액을 돌려드립니다.\n'
 '③ 계약의 무효, 효력상실, 해지 또는 소멸로 인하여 회사가 환급하여야 할 보험료가 있을'),
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
 'indexing': {'chunk_id': 'chunk_000103',
              'chunk_char_len': 275,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'normal',
              'tag_method': 'rule',
              'tag_confidence': 0.25}},
)
