from langchain_core.documents import Document

chunk = Document(
    page_content=('계약자가 연장된 보험계약을 취소하는 경우 회사는 최초연\n'
 '장된 날 이후 계약자가 납입한 보험료 전액을 환급합니다.\n'
 '\uf000 제6항에 따라 보험계약이 연장된 경우 보험계약의 연장\n'
 '일은 회사가 계약자의 재가입의사를 확인한 날(계약자 등이\n'
 '회사에 보험금을 청구함으로써 계약자에게 연락이 닿아 회\n'
 '사가 계약자의 재가입의사를 확인한 날 등)까지로 합니다.\n'
 '회사는 계약자 등이 회사에 보험금을 청구하는 등 계약자에\n'
 '게 연락이 닿으면 제4항의 내용과 90일 이내 계약자의 재가\n'
 '입의사가 확인되지 않는 경우 계약이 해지된다는 사실을 알'),
    metadata={'source_doc': {'total_pages': 160},
 'doc': {'doc_type': 'terms',
         'file_name': 'meritz_2_petpermint_cat_family_45_82_83_206.pdf',
         'insurer_code': 'meritz',
         'product_code': '2',
         'product_name': '무배당 펫퍼민트 Cat&Family보험 다이렉트2601',
         'total_pages': 1,
         'page': 1},
 'term_type': 'unknown',
 'clause': {'clause_type': 'other', 'risk_domains': ['digestive']},
 'indexing': {'chunk_id': 'chunk_000201',
              'chunk_char_len': 287,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'simple',
              'tag_method': 'rule',
              'tag_confidence': 0.25}},
)
