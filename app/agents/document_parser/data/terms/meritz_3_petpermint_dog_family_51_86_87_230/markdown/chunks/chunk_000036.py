from langchain_core.documents import Document

chunk = Document(
    page_content=('을 연단위 복리로 계산한 금액을 더하며, 나누어 지급할 금\n'
 '액을 일시에 지급하는 경우에는 평균공시이율을 연단위 복\n'
 '리로 할인한 금액을 지급합니다.# 【보험금 지급 예시】1. 일시에 지급할 금액을 나누어 지급하는 경우\n'
 '보험금 : 6천만원\n'
 '보험금 지급일자 : 2025년 4월 1일\n'
 '보험금을 일시에 지급받지 않고, 3년간 매년 동일한 금액\n'
 '으로 나누어 지급받는 경우| 지급일 | 보험금 받는 방법 변경 후 지급액 |\n'
 '| --- | --- |\n'
 '| 2025년 4월 1일 | 2천만원 |'),
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
