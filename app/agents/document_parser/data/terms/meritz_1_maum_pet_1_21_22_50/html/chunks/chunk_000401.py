from langchain_core.documents import Document

chunk = Document(
    page_content=('60%</td></tr><tr><td>6개월까지</td><td>연요율의 '
 '70%</td></tr><tr><td>7개월까지</td><td>연요율의 '
 '75%</td></tr><tr><td>8개월까지</td><td>연요율의 '
 '80%</td></tr><tr><td>9개월까지</td><td>연요율의 '
 '85%</td></tr><tr><td>10개월까지</td><td>연요율의 '
 '90%</td></tr><tr><td>11개월까지</td><td>연요율의 95%</td></tr></tbody></table><br><p '
 "id='93'"),
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
