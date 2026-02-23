from langchain_core.documents import Document

chunk = Document(
    page_content=('| --- | --- |\n'
 '| 보험기간 | 계약에 따라 보장을 받는 기간을 말합니다. |\n'
 '| 영업일 | 회사가 영업점에서 정상적으로 영업하는 날을 말하며, 토요일,‘관공서의 공휴일에 관한 규 정’에 따른 공휴일(대체공휴일 '
 '포함)과 근로 자의 날을 제외합니다. |\n'
 '# \uf000 보험료 관련 용어| 용어 | 정의 |\n'
 '| --- | --- |\n'
 '| 보험료 | 계약자가 매 납입기일에 납입하기로 한 보험료 로 기본계약 보험료와 특별약관이 부가된 경우 에는 특별약관 보험료의 합계액을 '
 '말합니다. |'),
    metadata={'source_doc': {'total_pages': 180},
 'doc': {'doc_type': 'terms',
         'file_name': 'meritz_3_petpermint_dog_family_51_86_87_230.pdf',
         'insurer_code': 'meritz',
         'product_code': '3',
         'product_name': '무배당 펫퍼민트 Puppy&Family보험 다이렉트2601',
         'total_pages': 1,
         'page': 1},
 'term_type': 'unknown',
 'clause': {'clause_type': 'exclusion', 'risk_domains': ['digestive']},
 'indexing': {'chunk_id': 'chunk_000153',
              'chunk_char_len': 268,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'simple',
              'tag_method': 'rule',
              'tag_confidence': 0.85}},
)
