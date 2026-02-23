from langchain_core.documents import Document

chunk = Document(
    page_content=('- 여야 합니다.\n'
 '- ⑥ 회사는 제1항에 따라 계약자를 변경한 경우, 변경된 계약자에게 보험증권 및 약관을 교\n'
 '- 부하고 변경된 계약자가 요청하는 경우 약관의 중요한 내용을 설명하여 드립니다.\n'
 '- 14 -제24조(계약의 소멸)# 반려동물의 사망 등으로 인하여 이 약관에서 규정하는 보험금 지급사유가 더 이상 발생할\n'
 '수 없는 경우에는 이 계약은 그 때부터 효력이 없습니다.제5관 보험료의 납입# 제25조(제1회 보험료 및 회사의 보장개시)- ① 회사는 '
 '계약의 청약을 승낙하고 제1회 보험료를 받은 때부터 이 약관이 정한 바에 따'),
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
