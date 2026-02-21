from langchain_core.documents import Document

chunk = Document(
    page_content=('| --- | --- |\n'
 '| 7일까지 | 연요율의 10% |\n'
 '| 15일까지 | 연요율의 15% |\n'
 '| 1개월까지 | 연요율의 20% |\n'
 '| 2개월까지 | 연요율의 30% |\n'
 '| 3개월까지 | 연요율의 40% |\n'
 '| 4개월까지 | 연요율의 50% |\n'
 '| 5개월까지 | 연요율의 60% |\n'
 '| 6개월까지 | 연요율의 70% |\n'
 '| 7개월까지 | 연요율의 75% |\n'
 '| 8개월까지 | 연요율의 80% |\n'
 '| 9개월까지 | 연요율의 85% |\n'
 '| 10개월까지 | 연요율의 90% |\n'
 '| 11개월까지 | 연요율의 95% |'),
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
