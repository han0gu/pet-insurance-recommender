from langchain_core.documents import Document

chunk = Document(
    page_content=('2조 제2호에 따른 전자서명이 있는 경우로서 상법 시행령 제44조의2에 정하는 바에 따\n'
 '라 본인 확인 및 위조ㆍ변조 방지에 대한 신뢰성을 갖춘 전자문서를 포함)으로 동의하\n'
 '여야 합니다.\n'
 '⑥ 회사는 제1항에 따라 계약자를 변경한 경우, 변경된 계약자에게 보험증권 및 약관을 교\n'
 '부하고 변경된 계약자가 요청하는 경우 약관의 중요한 내용을 설명하여 드립니다.- 14 -제24조(계약의 소멸)반려동물의 사망 등으로 '
 '인하여 이 약관에서 규정하는 보험금 지급사유가 더 이상 발생할'),
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
