from langchain_core.documents import Document

chunk = Document(
    page_content=('\uf000 회사는 계약자에게 재가입주기(보장내용 변경주기)가 끝 나는 날 이전까지 2회 이상 재가입 요건, 보장내용 변경내 역, 보험료 '
 '수준, 재가입 절차 및 재가입 의사 여부를 확인 하는 내용 등을 서면(등기우편 등), 전화(음성녹음), 전자 문서, 휴대전화 문자메시지 '
 '또는 이에 준하는 전자적 의사 표시 등으로 알려드리고, 회사는 계약자의 재가입의사를 전 화(음성녹음), 직접 방문 또는 전자적 의사표시, '
 '통신판매 계약의 경우 통신수단을 통해 확인합니다.'),
    metadata={'source_doc': {'total_pages': 180},
 'doc': {'doc_type': 'terms',
         'file_name': 'meritz_3_petpermint_dog_family_51_86_87_230.pdf',
         'insurer_code': 'meritz',
         'product_code': '3',
         'product_name': '무배당 펫퍼민트 Puppy&Family보험 다이렉트2601',
         'total_pages': 180,
         'page': 102},
 'term_type': 'special',
 'clause': {'clause_type': 'coverage', 'risk_domains': ['digestive']},
 'indexing': {'chunk_id': 'chunk_000248',
              'chunk_char_len': 252,
              'embedding_model': 'solar-embedding-1-large',
              'tag_method': 'rule',
              'tag_confidence': 0.8}},
)
