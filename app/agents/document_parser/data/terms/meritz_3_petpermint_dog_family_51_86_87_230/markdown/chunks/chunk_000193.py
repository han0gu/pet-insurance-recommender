from langchain_core.documents import Document

chunk = Document(
    page_content=('는 가입금액을 계약시 선택한 금액보다 적은 금액으로\n'
 '줄이는 것을 말합니다.(이에 따라 보험료, 보험금 및 해\n'
 '약환급금도 줄어듭니다)\uf000 계약자가 제2항에 따라 보험수익자를 변경하고자 할 경\n'
 '우 계약자와 피보험자가 동일하지 않을 때에는 보험금 지급\n'
 '사유가 발생하기 전에 피보험자가 서면(「전자서명법」 제2\n'
 '조 제2호에 따른 전자서명이 있는 경우로서 상법 시행령 제\n'
 '44조의2에 정하는 바에 따라 본인 확인 및 위조ㆍ변조 방지\n'
 '에 대한 신뢰성을 갖춘 전자문서를 포함)으로 동의하여야\n'
 '합니다.'),
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
