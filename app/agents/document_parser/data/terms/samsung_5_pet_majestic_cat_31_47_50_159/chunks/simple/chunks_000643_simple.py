from langchain_core.documents import Document

chunk = Document(
    page_content=('. 다만, 계약자가 재가입을 원 하지 않는 경우에는 해당 시점으로부터 계약은 해지됩니다(단, 최초연장된 날로부터 90일 이전에는 계약을 '
 '취소 또는 해지할 수 있습니다.) ⑩ 제7항 내지 제9항에 따라 계약이 해지된 경우 회사는 특별약관 일반사항 제35조(해 약 환급금) '
 '제1항에 따른 해약환급금을 계약자에게 지급합니다.'),
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
 'indexing': {'chunk_id': 'chunk_000643',
              'chunk_char_len': 178,
              'embedding_model': 'solar-embedding-1-large',
              'tag_method': 'llm',
              'tag_confidence': 0.95}},
)
