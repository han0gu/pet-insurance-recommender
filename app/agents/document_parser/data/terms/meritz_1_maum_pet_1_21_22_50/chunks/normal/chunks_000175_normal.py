from langchain_core.documents import Document

chunk = Document(
    page_content=('1. 계약자, 피보험자 또는 이들의 대리인이 고의 또는 중대한 과실로 보통약관 제15조 (계약 전 알릴 의무)를 위반하고 그 의무가 '
 '중요한 사항에 해당하는 경우 2. 뚜렷한 위험의 증가와 관련된 제15조(계약 후 알릴 의무) 제1항에서 정한 계약 후 알릴 의무를 계약자 '
 '또는 피보험자의 고의 또는 중대한 과실로 이행하지 않았을 때 3. 상당한 이유없이 손해조사를 거부 또는 회피할 때\n'
 '② 제1항 제1호에도 불구하고 다음 중 한가지의 경우에 해당되는 때에는 회사는 계약을 해지할 수 없습니다.'),
    metadata={'source_doc': {'total_pages': 50},
 'doc': {'doc_type': 'terms',
         'file_name': 'meritz_1_maum_pet_1_21_22_50.pdf',
         'insurer_code': 'meritz',
         'product_code': '1',
         'product_name': '메리츠 마음든든 반려동물보험',
         'total_pages': 50,
         'page': 28},
 'term_type': 'special',
 'clause': {'clause_type': 'exclusion', 'risk_domains': ['other']},
 'indexing': {'chunk_id': 'chunk_000175',
              'chunk_char_len': 274,
              'embedding_model': 'solar-embedding-1-large',
              'tag_method': 'llm',
              'tag_confidence': 0.95}},
)
