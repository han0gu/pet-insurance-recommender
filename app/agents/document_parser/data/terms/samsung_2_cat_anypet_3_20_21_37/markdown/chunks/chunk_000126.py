from langchain_core.documents import Document

chunk = Document(
    page_content=('- 추가 또는 환급보험료는 일단위로 계산하여 받거나 돌려드립니다.\n'
 '- ③ 회사는 제1항 및 제2항을 위반하였을 경우에 새로이 증가 또는 교체되는 해당피보험자에 대하여는\n'
 '- 보상하여 드리지 않습니다.\n'
 '- ④ 제1항에 따라 피보험자가 교체되는 경우에는 피보험자 교체 전 계약과 동일한 보장조건 및 인수기\n'
 '- 준에 따라 가입될 수 있으며, 피보험자 교체시점부터 잔여 보험기간(피보험자 교체 전 계약의 보험\n'
 '- 기간 만료일)까지 보상하여 드립니다.'),
    metadata={'source_doc': {'total_pages': 35},
 'doc': {'doc_type': 'terms',
         'file_name': 'samsung_2_cat_anypet_3_20_21_37.pdf',
         'insurer_code': 'samsung',
         'product_code': '2',
         'product_name': '(일반)반려묘보험 애니펫',
         'total_pages': 1,
         'page': 1},
 'term_type': 'unknown'},
)
