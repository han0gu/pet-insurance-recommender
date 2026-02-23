from langchain_core.documents import Document

chunk = Document(
    page_content=('⑨ 제5항에 따라 보험계약이 연장된 경우 계약자는 회사에 재가입 의사를 표시할 수 있\n'
 '습니다. 회사는 계약자의 재가입 의사가 확인되었을 때에는 제1항 및 제2항에서 정한\n'
 '절차에 따라 회사가 재가입 의사를 확인한 날에 판매중인 제2항의 반려동물보험 상품\n'
 '으로 재가입하는 것으로 하며, 기존 계약은 해지됩니다. 다만, 계약자가 재가입을 원\n'
 '하지 않는 경우에는 해당 시점으로부터 계약은 해지됩니다(단, 최초연장된 날로부터\n'
 '90일 이전에는 계약을 취소 또는 해지할 수 있습니다.)'),
    metadata={'source_doc': {'total_pages': 107},
 'doc': {'doc_type': 'terms',
         'file_name': 'samsung_4_pet_majestic_dog_33_49_52_139.pdf',
         'insurer_code': 'samsung',
         'product_code': '4',
         'product_name': '(장기)무배당 삼성화재 펫보험 위풍댕댕',
         'total_pages': 1,
         'page': 1},
 'term_type': 'unknown',
 'clause': {'clause_type': 'other', 'risk_domains': ['digestive']},
 'indexing': {'chunk_id': 'chunk_000546',
              'chunk_char_len': 264,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'simple',
              'tag_method': 'rule',
              'tag_confidence': 0.25}},
)
