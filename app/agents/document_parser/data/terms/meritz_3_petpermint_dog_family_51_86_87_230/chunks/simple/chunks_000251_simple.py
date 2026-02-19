from langchain_core.documents import Document

chunk = Document(
    page_content=('\uf000 제6항에 따라 보험계약이 연장된 경우 보험계약의 연장 일은 회사가 계약자의 재가입의사를 확인한 날(계약자 등이 회사에 '
 '보험금을 청구함으로써 계약자에게 연락이 닿아 회 사가 계약자의 재가입의사를 확인한 날 등)까지로 합니다. 회사는 계약자 등이 회사에 '
 '보험금을 청구하는 등 계약자에 게 연락이 닿으면 제4항의 내용과 90일 이내 계약자의 재가 입의사가 확인되지 않는 경우 계약이 해지된다는 '
 '사실을 알 려드립니다.'),
    metadata={'source_doc': {'total_pages': 180},
 'doc': {'doc_type': 'terms',
         'file_name': 'meritz_3_petpermint_dog_family_51_86_87_230.pdf',
         'insurer_code': 'meritz',
         'product_code': '3',
         'product_name': '무배당 펫퍼민트 Puppy&Family보험 다이렉트2601',
         'total_pages': 180,
         'page': 103},
 'term_type': 'special',
 'clause': {'clause_type': 'other', 'risk_domains': ['other']},
 'indexing': {'chunk_id': 'chunk_000251',
              'chunk_char_len': 231,
              'embedding_model': 'solar-embedding-1-large',
              'tag_method': 'llm',
              'tag_confidence': 0.95}},
)
