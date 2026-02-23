from langchain_core.documents import Document

chunk = Document(
    page_content=('- 가입의사를 확인한 날(계약자 등이 회사에 보험금을 청구함으로써 계약자에게 연\n'
 '- 락이 닿아 회사가 계약자의 재가입의사를 확인한 날 등)까지로 합니다. 회사는\n'
 '- 계약자 등이 회사에 보험금을 청구하는 등 계약자에게 연락이 닿으면 제3항의 내\n'
 '- 용과 90일 이내 계약자의 재가입의사가 확인되지 않는 경우 계약이 해지된다는\n'
 '- 사실을 알려드립니다.\n'
 '- \uf000 제7항에 따라 계약자에게 해지된다는 사실을 알려드린 최초시점부터 90일 이내에\n'
 '- 계약자의 재가입 의사가 확인되지 않는 경우 해당 시점부터 계약은 해지됩니다.'),
    metadata={'source_doc': {'total_pages': 113},
 'doc': {'doc_type': 'terms',
         'file_name': 'kb_17390_gold_pet_dog_54_72_74_166.pdf',
         'insurer_code': 'kb',
         'product_code': '17390',
         'product_name': '[일반보험] KB반려행복펫보험',
         'total_pages': 1,
         'page': 1},
 'term_type': 'unknown',
 'clause': {'clause_type': 'other', 'risk_domains': ['other']},
 'indexing': {'chunk_id': 'chunk_000539',
              'chunk_char_len': 287,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'simple',
              'tag_method': 'rule',
              'tag_confidence': 0.25}},
)
