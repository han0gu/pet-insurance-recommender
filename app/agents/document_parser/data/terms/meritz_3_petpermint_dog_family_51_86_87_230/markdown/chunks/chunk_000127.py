from langchain_core.documents import Document

chunk = Document(
    page_content=('약환급금 산출방법서」에 따라 계약자적립액에서 해당 중도\n'
 '인출금을 차감합니다.# 【보험연도】당해 연도 보험계약 해당일부터 차년도 보험계약 해당일\n'
 '전일까지 매1년 단위의 연도를 말합니다. 예를 들어, 보\n'
 '험계약일이 2023년 4월 1일인 경우 보험연도는 4월 1일\n'
 '부터 차년도 3월 31일까지 1년을 말합니다.82# 【중도인출금의 한도 예시】계약자가 요청한 시점에서 계산된 기본계약 해약환급금\n'
 '과 기본계약 적립부분 해약환급금 중 적은 금액이 100만\n'
 '원인 경우 중도인출 가능액은 80만원(100만원의 80%)이'),
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
