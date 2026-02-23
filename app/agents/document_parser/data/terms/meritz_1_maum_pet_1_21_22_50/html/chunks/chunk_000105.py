from langchain_core.documents import Document

chunk = Document(
    page_content=("id='3' style='font-size:14px'>- 11 -</footer><h1 id='4' "
 "style='font-size:14px'>제18조(사기에 의한 계약)</h1><br><p id='5' "
 "data-category='paragraph' style='font-size:14px'>계약자 또는 피보험자가 사기에 의하여 계약이 "
 '성립되었음을 회사가 증명하는 경우에는 계<br>약일부터 5년 이내(사기사실을 안 날부터 1개월 이내)에 계약을 취소할 수 '
 "있습니다.</p><p id='6' data-category='paragraph'"),
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
