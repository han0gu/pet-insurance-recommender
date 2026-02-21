from langchain_core.documents import Document

chunk = Document(
    page_content=("id='113' style='font-size:14px'>- 29 -</footer><h1 id='114' "
 "style='font-size:14px'>제20조(계약자의 임의해지)</h1><br><p id='115' "
 "data-category='paragraph' style='font-size:14px'>계약자는 손해가 발생하기 전에는 언제든지 계약을 "
 '해지할 수 있습니다'),
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
