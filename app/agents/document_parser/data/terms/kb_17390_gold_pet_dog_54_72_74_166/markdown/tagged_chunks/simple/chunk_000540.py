from langchain_core.documents import Document

chunk = Document(
    page_content=('- 계약자의 재가입 의사가 확인되지 않는 경우 해당 시점부터 계약은 해지됩니다.\n'
 '- \uf000 제5항에 따라 보험계약이 연장된 경우 계약자는 회사에 재가입 의사를 표시할 수\n'
 '- 있습니다. 회사는 계약자의 재가입 의사가 확인되었을 때에는 제1항 및 제2항에\n'
 '- 서 정한 절차에 따라 회사가 재가입 의사를 확인한 날에 판매중인 제2항의 반려동\n'
 '- 물보험 상품으로 재가입하는 것으로 하며, 기존 계약은 해지됩니다. 다만, 계약\n'
 '- 자가 재가입을 원하지 않는 경우에는 해당 시점으로부터 계약은 해지됩니다(단,'),
    metadata={'source_doc': {'total_pages': 113},
 'doc': {'doc_type': 'terms',
         'file_name': 'kb_17390_gold_pet_dog_54_72_74_166.pdf',
         'insurer_code': 'kb',
         'product_code': '17390',
         'product_name': '[일반보험] KB반려행복펫보험',
         'total_pages': 1,
         'page': 1},
 'term_type': 'unknown',
 'clause': {'clause_type': 'other', 'risk_domains': ['digestive']},
 'indexing': {'chunk_id': 'chunk_000540',
              'chunk_char_len': 276,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'simple',
              'tag_method': 'rule',
              'tag_confidence': 0.25}},
)
