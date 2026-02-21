from langchain_core.documents import Document

chunk = Document(
    page_content=("계약의 다른 조항에서 달리 정의되지 않는 한 다음<br>과 같습니다.</p><br><h1 id='6' "
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
