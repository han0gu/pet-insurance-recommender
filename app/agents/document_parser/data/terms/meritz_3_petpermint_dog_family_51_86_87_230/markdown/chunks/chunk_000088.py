from langchain_core.documents import Document

chunk = Document(
    page_content=('액하고자 할 때에는 그 감액된 부분은 해지된 것으로 보며,\n'
 '이로써 회사가 지급하여야 할 해약환급금이 있을 때에는 제\n'
 '35조(해약환급금) 제1항에 따른 해약환급금을 계약자에게\n'
 '지급합니다.# 【 감액 】보험료, 보험금, 계약자적립액 등을 산정하는 기준이 되\n'
 '는 가입금액을 계약시 선택한 금액보다 적은 금액으로\n'
 '줄이는 것을 말합니다.(이에 따라 보험료, 보험금 및 해\n'
 '약환급금도 줄어듭니다)\uf000 계약자가 제2항에 따라 보험수익자를 변경하고자 할 경\n'
 '우 계약자와 피보험자가 동일하지 않을 때에는 보험금 지급'),
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
