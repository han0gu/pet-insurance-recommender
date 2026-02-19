from langchain_core.documents import Document

chunk = Document(
    page_content=('- 1년 후 원리금 : 100원 + (100원×10%) = 110원 - 2년 후 원리금 : 110원 + (110원×10%) = 121원\n'
 '\uf000 기간과 날짜 관련 용어\n'
 '용어 | 정의\n'
 '보험기간 | 계약에 따라 보장을 받는 기간을 말합니다.\n'
 '영업일 | 회사가 영업점에서 정상적으로 영업하는 날을 말하며, 토요일,‘관공서의 공휴일에 관한 규 정’에 따른 공휴일(대체공휴일 포함)과 '
 '근로 자의 날을 제외합니다.\n'
 '\uf000 보험료 관련 용어\n'
 '용어 | 정의'),
    metadata={'source_doc': {'total_pages': 180},
 'doc': {'doc_type': 'terms',
         'file_name': 'meritz_3_petpermint_dog_family_51_86_87_230.pdf',
         'insurer_code': 'meritz',
         'product_code': '3',
         'product_name': '무배당 펫퍼민트 Puppy&Family보험 다이렉트2601',
         'total_pages': 180,
         'page': 53},
 'term_type': 'basic',
 'clause': {'clause_type': 'exclusion', 'risk_domains': ['digestive']},
 'indexing': {'chunk_id': 'chunk_000009',
              'chunk_char_len': 242,
              'embedding_model': 'solar-embedding-1-large',
              'tag_method': 'rule',
              'tag_confidence': 0.85}},
)
