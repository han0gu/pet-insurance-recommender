from langchain_core.documents import Document

chunk = Document(
    page_content=('- 절차에 따라 회사가 재가입 의사를 확인한 날에 판매중인 제2항의 반려동물보험 상품\n'
 '- 으로 재가입하는 것으로 하며, 기존 계약은 해지됩니다. 다만, 계약자가 재가입을 원\n'
 '- 하지 않는 경우에는 해당 시점으로부터 계약은 해지됩니다(단, 최초연장된 날로부터\n'
 '- 90일 이전에는 계약을 취소 또는 해지할 수 있습니다.)\n'
 '- ⑩ 제7항 내지 제9항에 따라 계약이 해지된 경우 회사는 특별약관 일반사항 제35조(해 약\n'
 '- 환급금) 제1항에 따른 해약환급금을 계약자에게 지급합니다.\n'
 '- 제28조 (준용규정)'),
    metadata={'source_doc': {'total_pages': 129},
 'doc': {'doc_type': 'terms',
         'file_name': 'samsung_5_pet_majestic_cat_31_47_50_159.pdf',
         'insurer_code': 'samsung',
         'product_code': '5',
         'product_name': '(장기)무배당 삼성화재 펫보험 의기냥냥',
         'total_pages': 1,
         'page': 1},
 'term_type': 'unknown',
 'clause': {'clause_type': 'other', 'risk_domains': ['digestive']},
 'indexing': {'chunk_id': 'chunk_000541',
              'chunk_char_len': 279,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'simple',
              'tag_method': 'rule',
              'tag_confidence': 0.25}},
)
