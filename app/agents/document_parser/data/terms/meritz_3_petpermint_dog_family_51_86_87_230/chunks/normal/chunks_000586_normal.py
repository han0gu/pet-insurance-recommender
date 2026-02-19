from langchain_core.documents import Document

chunk = Document(
    page_content=('용어 | 정의\n'
 '보험기간 | 계약에 따라 보장을 받는 기간을 말합니다.\n'
 '영업일 | 회사가 영업점에서 정상적으로 영업하는 날을 말하며, 토요일,‘관공서의 공휴일에 관한 규 정’에 따른 공휴일(대체공휴일 포함)과 '
 '근로 자의 날을 제외합니다.\n'
 '\uf000 보험료 관련 용어\n'
 '용어 | 정의\n'
 '보험료 | 계약에서 정한 손해를 보장하는데 필요한 보험 료를 말합니다.\n'
 '제3조(손해의 발생과 통지)\n'
 '\uf000 계약자 또는 피보험자는 아래와 같은 사실이 있는 경우 에는 지체없이 그 내용을 회사에 알려야 합니다.'),
    metadata={'source_doc': {'total_pages': 180},
 'doc': {'doc_type': 'terms',
         'file_name': 'meritz_3_petpermint_dog_family_51_86_87_230.pdf',
         'insurer_code': 'meritz',
         'product_code': '3',
         'product_name': '무배당 펫퍼민트 Puppy&Family보험 다이렉트2601',
         'total_pages': 180,
         'page': 176},
 'term_type': 'special',
 'clause': {'clause_type': 'exclusion', 'risk_domains': ['digestive']},
 'indexing': {'chunk_id': 'chunk_000586',
              'chunk_char_len': 266,
              'embedding_model': 'solar-embedding-1-large',
              'tag_method': 'rule',
              'tag_confidence': 0.85}},
)
