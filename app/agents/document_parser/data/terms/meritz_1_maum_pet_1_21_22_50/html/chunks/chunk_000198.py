from langchain_core.documents import Document

chunk = Document(
    page_content=('. 이 보험계약은 예금자보호법에 따라 해약<br>환급금(또는 만기 시 보험금)에 기타지급금을 합한 금액이 1인당 “1억원까지”(본 '
 '보험<br>회사의 여타 보호상품과 합산) 보호됩니다. 이와 별도로 본 보험회사 보호상품의 사고<br>보험금을 합산한 금액이 1인당 '
 '“1억원까지”보호됩니다'),
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
