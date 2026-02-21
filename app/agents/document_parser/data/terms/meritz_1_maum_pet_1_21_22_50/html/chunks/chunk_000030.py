from langchain_core.documents import Document

chunk = Document(
    page_content=("지급하지 않는 사유)</h1><br><p id='38' data-category='paragraph' "
 "style='font-size:14px'>① 회사는 다음 중 어느 한 가지로 보험금 지급사유가 발생한 때에는 보험금을 "
 "지급하지<br>않습니다.</p><br><p id='39' data-category='paragraph' "
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
