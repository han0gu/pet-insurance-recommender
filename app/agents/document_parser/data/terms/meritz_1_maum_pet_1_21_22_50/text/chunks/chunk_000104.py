from langchain_core.documents import Document

chunk = Document(
    page_content=('③ 계약의 무효, 효력상실, 해지 또는 소멸로 인하여 회사가 환급하여야 할 보험료가 있을\n'
 '때에는 계약자는 환급금을 청구하여야 하며, 회사는 청구일의 다음 날부터 지급일까지\n'
 '의 기간에 대하여 ‘보험개발원이 공시하는 보험계약대출이율’을 연단위 복리로 계산한\n'
 '금액을 더하여 지급합니다.【설명】보험사가 해지권을 행사하는 경우 위의 ‘청구일’은 보험사의 해지 의사표시\n'
 '(서면, 전자우편, 휴대전화 문자메시지 또는 이에 준하는 전자적 의사표시 포함)가 보험'),
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
