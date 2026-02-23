from langchain_core.documents import Document

chunk = Document(
    page_content=('. 한편 위험이 증가된 경우에는 납입보험<br>료의 증액 및 정산금액의 추가납입을 요구할 수 있으며, 계약자는 일시납 또는 잔여 '
 '보험<br>료 납입기간과 5년 중 큰 기간(단, 잔여 보험기간을 초과할 수 없음) 동안의 분납 중 선택<br>하여 정산금액을 납입하여야 '
 '합니다'),
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
