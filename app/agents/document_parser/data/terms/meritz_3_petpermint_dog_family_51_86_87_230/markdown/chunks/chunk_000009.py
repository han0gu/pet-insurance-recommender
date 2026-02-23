from langchain_core.documents import Document

chunk = Document(
    page_content=('로 하는 이자 계산방법을 말합니다.\n'
 '원금 100원, 이자율 연 10%를 가정할 때- - 1년 후 원리금 : 100원 + (100원×10%) = 110원\n'
 '- - 2년 후 원리금 : 110원 + (110원×10%) = 121원\n'
 '# \uf000 기간과 날짜 관련 용어| 용어 | 정의 |\n'
 '| --- | --- |\n'
 '| 보험기간 | 계약에 따라 보장을 받는 기간을 말합니다. |\n'
 '| 영업일 | 회사가 영업점에서 정상적으로 영업하는 날을 말하며, 토요일,‘관공서의 공휴일에 관한 규 정’에 따른 공휴일(대체공휴일 '
 '포함)과 근로 자의 날을 제외합니다. |'),
    metadata={'source_doc': {'total_pages': 180},
 'doc': {'doc_type': 'terms',
         'file_name': 'meritz_3_petpermint_dog_family_51_86_87_230.pdf',
         'insurer_code': 'meritz',
         'product_code': '3',
         'product_name': '무배당 펫퍼민트 Puppy&Family보험 다이렉트2601',
         'total_pages': 1,
         'page': 1},
 'term_type': 'unknown'},
)
