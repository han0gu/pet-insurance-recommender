from langchain_core.documents import Document

chunk = Document(
    page_content=('입의사가 확인되지 않는 경우 계약이 해지된다는 사실을 알\n'
 '려드립니다.\uf000 제8항에 따라 계약자에게 해지된다는 사실을 알려드린\n'
 '최초시점부터 90일 이내에 계약자의 재가입 의사가 확인되\n'
 '지 않는 경우 해당 시점부터 계약은 해지됩니다.\uf000 제6항에 따라 보험계약이 연장된 경우 계약자는 회사에\n'
 '\uf000\n'
 '재가입 의사를 표시할 수 있습니다. 회사는 계약자의 재가\n'
 '입 의사가 확인되었을 때에는 제2항 및 제3항에서 정한 절\n'
 '차에 따라 회사가 재가입 의사를 확인한 날에 판매중인 제3\n'
 '항의 반려동물보험 상품으로 재가입하는 것으로 하며, 기존'),
    metadata={'source_doc': {'total_pages': 180},
 'doc': {'doc_type': 'terms',
         'file_name': 'meritz_3_petpermint_dog_family_51_86_87_230.pdf',
         'insurer_code': 'meritz',
         'product_code': '3',
         'product_name': '무배당 펫퍼민트 Puppy&Family보험 다이렉트2601',
         'total_pages': 1,
         'page': 1},
 'term_type': 'unknown',
 'clause': {'clause_type': 'other', 'risk_domains': ['digestive']},
 'indexing': {'chunk_id': 'chunk_000203',
              'chunk_char_len': 290,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'simple',
              'tag_method': 'rule',
              'tag_confidence': 0.25}},
)
