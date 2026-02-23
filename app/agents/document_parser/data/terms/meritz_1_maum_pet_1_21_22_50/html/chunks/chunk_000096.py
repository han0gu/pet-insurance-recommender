from langchain_core.documents import Document

chunk = Document(
    page_content=("중 하나에 해당하는 경우에는 회사는 계약을<br>해지할 수 없습니다.</p><br><p id='1' data-category='list' "
 "style='font-size:14px'>1"),
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
