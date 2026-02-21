from langchain_core.documents import Document

chunk = Document(
    page_content=('사는 이를 거절할 수 없습니다. 다만, 재가입 계약이 직전\n'
 '계약보다 보장내용 및 범위 등이 확대된 경우 확대된 내용\n'
 '에 대해 회사는 재가입 시점의 인수기준에 따라 승낙하거나\n'
 '일부 보장을 제한할 수 있습니다.\uf000 회사는 계약자에게 재가입주기(보장내용 변경주기)가 끝\n'
 '나는 날 이전까지 2회 이상 재가입 요건, 보장내용 변경내\n'
 '역, 보험료 수준, 재가입 절차 및 재가입 의사 여부를 확인\n'
 '하는 내용 등을 서면(등기우편 등), 전화(음성녹음), 전자\n'
 '문서, 휴대전화 문자메시지 또는 이에 준하는 전자적 의사'),
    metadata={'source_doc': {'total_pages': 180},
 'doc': {'doc_type': 'terms',
         'file_name': 'meritz_3_petpermint_dog_family_51_86_87_230.pdf',
         'insurer_code': 'meritz',
         'product_code': '3',
         'product_name': '무배당 펫퍼민트 Puppy&Family보험 다이렉트2601',
         'total_pages': 1,
         'page': 1},
 'term_type': 'unknown',
 'clause': {'clause_type': 'coverage', 'risk_domains': ['digestive']},
 'indexing': {'chunk_id': 'chunk_000199',
              'chunk_char_len': 279,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'simple',
              'tag_method': 'rule',
              'tag_confidence': 0.8}},
)
