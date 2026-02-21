from langchain_core.documents import Document

chunk = Document(
    page_content=("id='26' style='font-size:14px'>- 13 -</footer><h1 id='27' "
 "style='font-size:14px'>제22조(계약의 무효)</h1><br><p id='28' "
 "data-category='paragraph' style='font-size:14px'>회사는 다음의 경우에는 계약을 무효로 하며 이미 "
 '납입한 보험료를 돌려드립니다'),
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
