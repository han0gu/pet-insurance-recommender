from langchain_core.documents import Document

chunk = Document(
    page_content=('- 연장된 날 이후 계약자가 납입한 보험료 전액을 환급합니다.\n'
 '- ⑦ 제5항에 따라 보험계약이 연장된 경우 보험계약의 연장일은 회사가 계약자의 재가입\n'
 '- 의사를 확인한 날(계약자 등이 회사에 보험금을 청구함으로써 계약자에게 연락이 닿\n'
 '- 아 회사가 계약자의 재가입의사를 확인한 날 등)까지로 합니다. 회사는 계약자 등이\n'
 '- 회사에 보험금을 청구하는 등 계약자에게 연락이 닿으면 제3항의 내용과 90일 이내\n'
 '- 계약자의 재가입의사가 확인되지 않는 경우 계약이 해지된다는 사실을 알려드립니다.'),
    metadata={'source_doc': {'total_pages': 129},
 'doc': {'doc_type': 'terms',
         'file_name': 'samsung_5_pet_majestic_cat_31_47_50_159.pdf',
         'insurer_code': 'samsung',
         'product_code': '5',
         'product_name': '(장기)무배당 삼성화재 펫보험 의기냥냥',
         'total_pages': 1,
         'page': 1},
 'term_type': 'unknown',
 'clause': {'clause_type': 'renewal', 'risk_domains': ['digestive']},
 'indexing': {'chunk_id': 'chunk_000539',
              'chunk_char_len': 273,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'normal',
              'tag_method': 'rule',
              'tag_confidence': 0.55}},
)
