from langchain_core.documents import Document

chunk = Document(
    page_content=('. 다만, 회사는 법정상속<br>인이 보험수익자로 지정된 경우에는 제1항의 통지를 계약자에게 할 수 있습니다.<br>④ 회사는 제1항의 '
 '통지를 계약이 해지된 날부터 7일 이내에 하여야 합니다'),
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
