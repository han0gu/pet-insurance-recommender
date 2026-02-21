from langchain_core.documents import Document

chunk = Document(
    page_content=('. 보험증권에 기재된 피보험자(이하「피보험자 본인」이라 합니다)<br>2. 피보험자 본인의 가족관계등록상 또는 주민등록상에 기재된 '
 '배우자(이하「배우자」<br>라 합니다)<br>3'),
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
