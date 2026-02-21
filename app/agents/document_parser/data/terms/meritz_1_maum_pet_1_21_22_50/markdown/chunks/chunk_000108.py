from langchain_core.documents import Document

chunk = Document(
    page_content=('- 실 또는 소멸의 원인이 생긴 날 또는 해지일이 속하는 보험연도의 보험료는 제1항의\n'
 '- 규정을 적용하고 그 이후의 보험연도에 속하는 보험료는 전액을 돌려드립니다.\n'
 '- ③ 계약의 무효, 효력상실, 해지 또는 소멸로 인하여 회사가 환급하여야 할 보험료가 있을\n'
 '- 때에는 계약자는 환급금을 청구하여야 하며, 회사는 청구일의 다음 날부터 지급일까지\n'
 '- 의 기간에 대하여 ‘보험개발원이 공시하는 보험계약대출이율’을 연단위 복리로 계산한\n'
 '- 금액을 더하여 지급합니다.'),
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
