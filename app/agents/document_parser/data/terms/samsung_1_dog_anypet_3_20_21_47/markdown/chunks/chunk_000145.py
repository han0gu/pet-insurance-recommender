from langchain_core.documents import Document

chunk = Document(
    page_content=('- 증가 또는 교체되는 보험의 목적의 보험기간은 이 계약의 남은 보험기간으로 하고, 이로 인하여 발\n'
 '- 생되는 추가 또는 환급보험료는 일단위로 계산하여 받거나 돌려 드립니다.\n'
 '- ③ 회사는 제1항 및 제2항을 위반하였을 경우에 새로이 증가 또는 교체되는 해당 보험의 목적에 대하\n'
 '- 여는 보상하여 드리지 않습니다.\n'
 '- ④ 제1항에 따라 보험의 목적이 교체되는 경우에는 보험의 목적 교체전 계약과 동일한 보장조건 및\n'
 '- 인수기준에 따라 가입될 수 있으며, 보험의 목적 교체시점부터 잔여 보험기간(보험의 목적 교체전'),
    metadata={'source_doc': {'total_pages': 45},
 'doc': {'doc_type': 'terms',
         'file_name': 'samsung_1_dog_anypet_3_20_21_47.pdf',
         'insurer_code': 'samsung',
         'product_code': '1',
         'product_name': '(일반)반려견보험 애니펫',
         'total_pages': 1,
         'page': 1},
 'term_type': 'unknown'},
)
