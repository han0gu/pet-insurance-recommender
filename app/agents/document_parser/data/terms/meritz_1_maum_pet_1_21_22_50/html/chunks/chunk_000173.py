from langchain_core.documents import Document

chunk = Document(
    page_content=("id='87' style='font-size:14px'>제32조(회사의 파산선고와 해지)</h1><br><p id='88' "
 "data-category='list' style='font-size:14px'>① 회사가 파산의 선고를 받은 때에는 계약자는 계약을 "
 '해지할 수 있습니다.<br>② 제1항의 규정에 따라 해지하지 않은 계약은 파산선고 후 3개월이 지난 때에는 그 효력<br>을 '
 "잃습니다.<br>③ 제1항의 규정에 따라 계약이 해지되거나 제2항의 규정에 따라 계약이 효력을 잃는 경</p><footer id='89'"),
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
