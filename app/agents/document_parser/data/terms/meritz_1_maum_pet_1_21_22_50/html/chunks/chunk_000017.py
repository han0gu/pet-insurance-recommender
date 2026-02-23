from langchain_core.documents import Document

chunk = Document(
    page_content=("style='font-size:14px'>- 2 -</footer><h1 id='25' "
 "style='font-size:18px'>제4조(보험금의 지급사유)</h1><br><p id='26' "
 "data-category='list' style='font-size:18px'>① 회사는 보험기간 중에 보험증권에 기재된 반려동물에게 "
 '질병 또는 상해가 발생하여 그<br>치료를 직접적인 목적으로 동물병원에 통원 또는 입원하여 수의사에게 치료를 받은 때<br>에는 '
 '피보험자가 부담한 반려동물의 치료비를 이 약관에 따라 피보험자에게'),
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
