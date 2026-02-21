from langchain_core.documents import Document

chunk = Document(
    page_content=('- 계약 후 알릴 의무를 계약자 또는 피보험자의 고의 또는 중대한 과실로 이행하지\n'
 '- 않았을 때\n'
 '# ② 제1항 제1호의 경우에도 불구하고 다음 중 하나에 해당하는 경우에는 회사는 특별약\n'
 '관을 해지할 수 없습니다.- 1. 회사가 최초계약 체결당시에 그 사실을 알았거나 과실로 인하여 알지 못하였을 때\n'
 '- 2. 회사가 그 사실을 안 날부터 1개월 이상 지났거나 또는 제1회 보험료를 받은 때부\n'
 '- 터 보험금 지급사유가 발생하지 않고 2년(진단계약의 경우 질병에 대하여는 1년)\n'
 '- 이 지났을 때'),
    metadata={'source_doc': {'total_pages': 107},
 'doc': {'doc_type': 'terms',
         'file_name': 'samsung_4_pet_majestic_dog_33_49_52_139.pdf',
         'insurer_code': 'samsung',
         'product_code': '4',
         'product_name': '(장기)무배당 삼성화재 펫보험 위풍댕댕',
         'total_pages': 1,
         'page': 1},
 'term_type': 'unknown',
 'clause': {'clause_type': 'coverage', 'risk_domains': ['other']},
 'indexing': {'chunk_id': 'chunk_000197',
              'chunk_char_len': 274,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'simple',
              'tag_method': 'rule',
              'tag_confidence': 0.8}},
)
