from langchain_core.documents import Document

chunk = Document(
    page_content=('- 으로 하고, 이로 인하여 발생되는 추가 또는 환급보험료는 일단위로 계산하여 받거나\n'
 '- 돌려 드립니다.\n'
 '- ③ 회사는 제1항 및 제2항을 위반하였을 경우에 새로이 증가 또는 교체되는 해당 보험의\n'
 '- 목적에 대하여는 보상하여 드리지 않습니다.\n'
 '- ④ 제1항에 따라 보험의 목적이 교체되는 경우에는 보험의 목적 교체전 계약과 동일한 보\n'
 '- 장조건 및 인수기준에 따라 가입될 수 있으며, 보험의 목적 교체시점부터 잔여 보험기\n'
 '- 간(보험의 목적 교체전 계약의 보험기간 만료일)까지 보상하여 드립니다.'),
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
