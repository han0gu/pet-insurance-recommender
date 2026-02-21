from langchain_core.documents import Document

chunk = Document(
    page_content=('피보험자가 다음 중 한가지의 단체에 소속되어야 하며, 단체를 대표하여 계약자로 된 자<br>가 단체보험 계약상의 모든 권리, 의무를 '
 "행사할 수 있어야 합니다.</p><br><h1 id='51' style='font-size:14px'>1"),
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
