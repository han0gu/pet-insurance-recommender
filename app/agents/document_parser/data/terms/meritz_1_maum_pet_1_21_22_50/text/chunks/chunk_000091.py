from langchain_core.documents import Document

chunk = Document(
    page_content=('17조(알릴 의무 위반의 효과), 제18조(사기에 의한 계약), 제19조(보험계약의 성립) 및\n'
 '제25조(제1회 보험료 및 회사의 보장개시)의 규정을 준용합니다. 이 때 회사는 해지 전\n'
 '발생한 보험금 지급사유를 이유로 부활(효력회복)을 거절하지 않습니다.\n'
 '③ 제1항에서 정한 계약의 부활이 이루어진 경우라도 계약자 또는 피보험자가 최초 계약\n'
 '청약시(2회 이상 부활이 이루어진 경우 종전 모든 부활 청약 포함) 제15조(계약 전 알'),
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
