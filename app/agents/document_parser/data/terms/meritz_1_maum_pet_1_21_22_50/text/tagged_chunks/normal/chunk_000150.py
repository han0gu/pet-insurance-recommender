from langchain_core.documents import Document

chunk = Document(
    page_content=('(계약 전 알릴 의무)를 위반하고 그 의무가 중요한 사항에 해당하는 경우\n'
 '2. 뚜렷한 위험의 증가와 관련된 제15조(계약 후 알릴 의무) 제1항에서 정한 계약 후\n'
 '알릴 의무를 계약자 또는 피보험자의 고의 또는 중대한 과실로 이행하지 않았을 때\n'
 '3. 상당한 이유없이 손해조사를 거부 또는 회피할 때② 제1항 제1호에도 불구하고 다음 중 한가지의 경우에 해당되는 때에는 회사는 '
 '계약을\n'
 '해지할 수 없습니다.1. 회사가 최초계약 체결당시에 그 사실을 알았거나 과실로 알지 못하였을 때'),
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
 'indexing': {'chunk_id': 'chunk_000150',
              'chunk_char_len': 266,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'normal',
              'tag_method': 'rule',
              'tag_confidence': 0.25}},
)
