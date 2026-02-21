from langchain_core.documents import Document

chunk = Document(
    page_content=('- 2. 이 계약에서 보장하는 위험과 동일한 위험을 보장하는 계약을 다른 보험자와 체결하\n'
 '- 27 -고자 할 때 또는 이와 같은 계약이 있음을 알았을 때3. 위험이 뚜렷이 변경되거나 변경되었음을 알았을 때- ② 회사는 제1항에 '
 '따라 위험이 감소된 경우에는 그 차액보험료를 돌려드리며, 위험이 증\n'
 '- 가된 경우에는 통지를 받은 날부터 1개월 이내에 보험료의 증액을 청구하거나 계약을\n'
 '- 해지할 수 있습니다.\n'
 '- ③ 계약자 또는 피보험자는 주소 또는 연락처가 변경된 경우에는 지체없이 이를 회사에 알'),
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
