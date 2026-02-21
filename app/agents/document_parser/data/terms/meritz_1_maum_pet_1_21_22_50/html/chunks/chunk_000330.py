from langchain_core.documents import Document

chunk = Document(
    page_content=('. 회사는 보험기간 만료 후 보험의 목적의 정보의 변경에 따라 산출된 확정보험료와 계약<br>을 체결할 때 산출한 예치보험료를 비교하여 '
 '그 차액을 정산합니다.<br>4'),
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
