from langchain_core.documents import Document

chunk = Document(
    page_content=('. 회사는 계약자 등이 회사에 보험금을 청구하는 등 계약자에게 연락이 닿으면 제3항의 내용과 90일 이내 계약자의 재가입의사가 확인되지 '
 '않는 경우 계약이 해지된다는 사실을 알려드립니다. ⑧ 제7항에 따라 계약자에게 해지된다는 사실을 알려드린 최초시점부터 90일 이내에 계 '
 '약자의 재가입 의사가 확인되지 않는 경우 해당 시점부터 계약은 해지됩니다.'),
    metadata={'source_doc': {'total_pages': 129},
 'doc': {'doc_type': 'terms',
         'file_name': 'samsung_5_pet_majestic_cat_31_47_50_159.pdf',
         'insurer_code': 'samsung',
         'product_code': '5',
         'product_name': '(장기)무배당 삼성화재 펫보험 의기냥냥',
         'total_pages': 107,
         'page': 106},
 'term_type': 'special',
 'clause': {'clause_type': 'other', 'risk_domains': ['other']},
 'indexing': {'chunk_id': 'chunk_000641',
              'chunk_char_len': 193,
              'embedding_model': 'solar-embedding-1-large',
              'tag_method': 'llm',
              'tag_confidence': 0.95}},
)
