from langchain_core.documents import Document

chunk = Document(
    page_content=('제9조(의무보험과의 관계)\n'
 '① 회사는 이 특별약관에 의하여 보상하여야 하는 금액이 의무보험에서 보상하는 금액을 초과할 때에 한하여 그 초과액만을 보상합니다. 다만, '
 '의무보험이 다수인 경우에는 제 10조(보험금의 분담)를 따릅니다. ② 제1항의 의무보험은 피보험자가 법률에 의하여 의무적으로 가입하여야 '
 '하는 보험으로 서 공제계약을 포함합니다.\n'
 '【공제계약】'),
    metadata={'source_doc': {'total_pages': 50},
 'doc': {'doc_type': 'terms',
         'file_name': 'meritz_1_maum_pet_1_21_22_50.pdf',
         'insurer_code': 'meritz',
         'product_code': '1',
         'product_name': '메리츠 마음든든 반려동물보험',
         'total_pages': 50,
         'page': 25},
 'term_type': 'special',
 'clause': {'clause_type': 'coverage', 'risk_domains': ['other']},
 'indexing': {'chunk_id': 'chunk_000154',
              'chunk_char_len': 197,
              'embedding_model': 'solar-embedding-1-large',
              'tag_method': 'llm',
              'tag_confidence': 0.85}},
)
