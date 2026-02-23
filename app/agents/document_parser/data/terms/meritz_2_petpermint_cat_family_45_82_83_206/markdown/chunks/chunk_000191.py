from langchain_core.documents import Document

chunk = Document(
    page_content=('때에는 회사의 사업방법서에서 정하는 방법에 따라 이를 변\n'
 '경하여 드립니다.\n'
 '\uf000 회사는 계약자가 제1항 제5호에 따라 보험가입금액을 감\n'
 '액하고자 할 때에는 그 감액된 부분은 해지된 것으로 보며,\n'
 '이로써 회사가 지급하여야 할 해약환급금이 있을 때에는 보\n'
 '통약관 제35조(해약환급금) 제1항에 따른 해약환급금을 계\n'
 '약자에게 지급합니다.# 【감액】보험료, 보험금, 계약자적립액 등을 산정하는 기준이 되\n'
 '는 가입금액을 계약시 선택한 금액보다 적은 금액으로\n'
 '줄이는 것을 말합니다.(이에 따라 보험료, 보험금 및 해'),
    metadata={'source_doc': {'total_pages': 160},
 'doc': {'doc_type': 'terms',
         'file_name': 'meritz_2_petpermint_cat_family_45_82_83_206.pdf',
         'insurer_code': 'meritz',
         'product_code': '2',
         'product_name': '무배당 펫퍼민트 Cat&Family보험 다이렉트2601',
         'total_pages': 1,
         'page': 1},
 'term_type': 'unknown'},
)
