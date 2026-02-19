from langchain_core.documents import Document

chunk = Document(
    page_content=('【보험금 지급 예시】\n'
 '1. 일시에 지급할 금액을 나누어 지급하는 경우 보험금 : 6천만원 보험금 지급일자 : 2025년 4월 1일 보험금을 일시에 지급받지 '
 '않고, 3년간 매년 동일한 금액 으로 나누어 지급받는 경우\n'
 '지급일 | 보험금 받는 방법 변경 후 지급액\n'
 '2025년 4월 1일 | 2천만원\n'
 '2026년 4월 1일 | 2천만원 × (1 + 평균공시이율)\n'
 '2027년 4월 1일 | 2천만원 × (1 + 평균공시이율)2'),
    metadata={'source_doc': {'total_pages': 180},
 'doc': {'doc_type': 'terms',
         'file_name': 'meritz_3_petpermint_dog_family_51_86_87_230.pdf',
         'insurer_code': 'meritz',
         'product_code': '3',
         'product_name': '무배당 펫퍼민트 Puppy&Family보험 다이렉트2601',
         'total_pages': 180,
         'page': 60},
 'term_type': 'basic',
 'clause': {'clause_type': 'coverage', 'risk_domains': ['other']},
 'indexing': {'chunk_id': 'chunk_000044',
              'chunk_char_len': 232,
              'embedding_model': 'solar-embedding-1-large',
              'tag_method': 'rule',
              'tag_confidence': 0.8}},
)
