from langchain_core.documents import Document

chunk = Document(
    page_content=('. 「소득세법 제59조의4(특별세액공제) 제1항 제2호」에 따라 보험료가 특별세액공제<br>의 대상이 되는 보험</p><h1 '
 "id='36' style='font-size:14px'>【소득세법 제59조의4(특별세액공제)】</h1><br><p id='37' "
 "data-category='paragraph' style='font-size:14px'>① 근로소득이 있는 거주자(일용근로자는 제외한다"),
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
