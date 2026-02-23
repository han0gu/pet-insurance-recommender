from langchain_core.documents import Document

chunk = Document(
    page_content=('| 해약 환급금 | 계약이 해지되는 때에 회사가 계약자에게 돌려 주는 금액을 말합니다. |\n'
 '# 【연단위 복리】회사가 지급할 금전에 이자를 줄 때, 1년마다 마지막 날\n'
 '에 그 이자를 원금에 더한 금액을 다음 1년의 원금으로\n'
 '하는 이자 계산방법을 말합니다.\n'
 '원금 100원, 이자율 연 10%를 가정할 때- - 1년 후 원리금 : 100원 + (100원×10%) = 110원\n'
 '- - 2년 후 원리금 : 110원 + (110원×10%) = 121원\n'
 '\uf000 기간 관련 용어175| 용어 | 정의 |\n'
 '| --- | --- |'),
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
