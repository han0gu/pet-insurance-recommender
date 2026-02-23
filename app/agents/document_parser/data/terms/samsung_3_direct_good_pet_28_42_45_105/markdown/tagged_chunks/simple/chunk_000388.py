from langchain_core.documents import Document

chunk = Document(
    page_content=('- 회사에 보험금을 청구하는 등 계약자에게 연락이 닿으면 제3항의 내용과 90일 이내\n'
 '- 계약자의 재가입의사가 확인되지 않는 경우 계약이 해지된다는 사실을 알려드립니다.\n'
 '- ⑧ 제7항에 따라 계약자에게 해지된다는 사실을 알려드린 최초시점부터 90일 이내에 계\n'
 '- 약자의 재가입 의사가 확인되지 않는 경우 해당 시점부터 계약은 해지됩니다.\n'
 '- ⑨ 제5항에 따라 보험계약이 연장된 경우 계약자는 회사에 재가입 의사를 표시할 수 있\n'
 '- 습니다. 회사는 계약자의 재가입 의사가 확인되었을 때에는 제1항 및 제2항에서 정한'),
    metadata={'source_doc': {'total_pages': 78},
 'doc': {'doc_type': 'terms',
         'file_name': 'samsung_3_direct_good_pet_28_42_45_105.pdf',
         'insurer_code': 'samsung',
         'product_code': '3',
         'product_name': '(장기)무배당 삼성화재 다이렉트 '
                         '착한펫보험(강아지)',
         'total_pages': 1,
         'page': 1},
 'term_type': 'unknown',
 'clause': {'clause_type': 'other', 'risk_domains': ['digestive']},
 'indexing': {'chunk_id': 'chunk_000388',
              'chunk_char_len': 287,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'simple',
              'tag_method': 'rule',
              'tag_confidence': 0.25}},
)
