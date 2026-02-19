from langchain_core.documents import Document

chunk = Document(
    page_content=('. 회사는 계약자의 재가입 의사가 확인되었을 때에는 제1항 및 제2항에서 정한 절차에 따라 회사가 재가입 의사를 확인한 날에 판매중인 '
 '제2항의 반려동물보험 상품 으로 재가입하는 것으로 하며, 기존 계약은 해지됩니다. 다만, 계약자가 재가입을 원 하지 않는 경우에는 해당 '
 '시점으로부터 계약은 해지됩니다(단, 최초연장된 날로부터 90일 이전에는 계약을 취소 또는 해지할 수 있습니다.) ⑩ 제7항 내지 제9항에 '
 '따라 계약이 해지된 경우 회사는 특별약관 일반사항 제35조(해약 환급금) 제1항에 따른 해약환급금을 계약자에게 지급합니다.'),
    metadata={'source_doc': {'total_pages': 78},
 'doc': {'doc_type': 'terms',
         'file_name': 'samsung_3_direct_good_pet_28_42_45_105.pdf',
         'insurer_code': 'samsung',
         'product_code': '3',
         'product_name': '(장기)무배당 삼성화재 다이렉트 '
                         '착한펫보험(강아지)',
         'total_pages': 66,
         'page': 76},
 'term_type': 'special',
 'clause': {'clause_type': 'other', 'risk_domains': ['other']},
 'indexing': {'chunk_id': 'chunk_000452',
              'chunk_char_len': 296,
              'embedding_model': 'solar-embedding-1-large',
              'tag_method': 'llm',
              'tag_confidence': 0.95}},
)
