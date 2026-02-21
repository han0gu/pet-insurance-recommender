from langchain_core.documents import Document

chunk = Document(
    page_content=('험금을 지급합니다.피보험자가 부담한 총 '
 '비용금액$$\\overline{{{\\bf1}}}|\\stackrel{\\cong}{\\longrightarrow}\\overline{{{\\bf'),
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
