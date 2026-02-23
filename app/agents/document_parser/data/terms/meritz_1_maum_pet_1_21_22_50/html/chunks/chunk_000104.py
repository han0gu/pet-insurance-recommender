from langchain_core.documents import Document

chunk = Document(
    page_content=('보험금을 지<br>급합니다.<br>⑦ 회사는 다른 보험가입내역에 대한 계약 전 알릴 의무 위반을 이유로 계약을 해지하거나<br>보험금 '
 '지급을 거절하지 않습니다.<br>⑧ 제28조(보험료의 납입을 연체하여 해지된 계약의 부활(효력회복))에 따라 이 계약이 부<br>활이 '
 '이루어진 경우에는 부활계약을 제2항의 최초계약으로 봅니다.(부활(효력회복)이<br>여러차례 발생된 경우에는 각각의 부활(효력회복)계약을 '
 "최초계약으로 봅니다)</p><footer id='3' style='font-size:14px'>- 11 -</footer><h1"),
    metadata={'source_doc': {'total_pages': 50},
 'doc': {'doc_type': 'terms',
         'file_name': 'meritz_1_maum_pet_1_21_22_50.pdf',
         'insurer_code': 'meritz',
         'product_code': '1',
         'product_name': '메리츠 마음든든 반려동물보험',
         'total_pages': 1,
         'page': 1},
 'term_type': 'unknown'},
)
