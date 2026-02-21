from langchain_core.documents import Document

chunk = Document(
    page_content=('10%</td></tr><tr><td>15일까지</td><td>연요율의 '
 '15%</td></tr><tr><td>1개월까지</td><td>연요율의 '
 '20%</td></tr><tr><td>2개월까지</td><td>연요율의 '
 '30%</td></tr><tr><td>3개월까지</td><td>연요율의 '
 '40%</td></tr><tr><td>4개월까지</td><td>연요율의 '
 '50%</td></tr><tr><td>5개월까지</td><td>연요율의 '
 '60%</td></tr><tr><td>6개월까지</td><td>연요율의'),
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
