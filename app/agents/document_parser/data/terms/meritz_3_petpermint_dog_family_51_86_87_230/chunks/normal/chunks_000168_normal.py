from langchain_core.documents import Document

chunk = Document(
    page_content=('② 관련 법률의 개정 또는 폐지 등에 따라 약관에서 정한 보험금 지급사유의 판정이 불가능한 경우 ③ 관련 법률의 개정 또는 폐지 등에 '
 '따라 계약유지 필요 가 없어지는 경우 ④ 기타 금융위원회 등의 명령이 있는 경우'),
    metadata={'source_doc': {'total_pages': 180},
 'doc': {'doc_type': 'terms',
         'file_name': 'meritz_3_petpermint_dog_family_51_86_87_230.pdf',
         'insurer_code': 'meritz',
         'product_code': '3',
         'product_name': '무배당 펫퍼민트 Puppy&Family보험 다이렉트2601',
         'total_pages': 180,
         'page': 85},
 'term_type': 'basic',
 'clause': {'clause_type': 'coverage', 'risk_domains': ['digestive']},
 'indexing': {'chunk_id': 'chunk_000168',
              'chunk_char_len': 119,
              'embedding_model': 'solar-embedding-1-large',
              'tag_method': 'rule',
              'tag_confidence': 0.55}},
)
