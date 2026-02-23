from langchain_core.documents import Document

chunk = Document(
    page_content=("id='61' data-category='paragraph' style='font-size:14px'>약정된 기일까지 보험료가 납입되지 "
 "않을 경우, 회사가 계약자에게 납입을 재촉하<br>는 것을 말합니다.</p><h1 id='62' "
 "style='font-size:14px'>제28조(보험료의 납입을 연체하여 해지된 계약의 부활(효력회복))</h1><br><p "
 "id='63' data-category='paragraph' style='font-size:14px'>① 제27조(보험료의 납입이 "
 '연체되는 경우 납입최고(독촉)와 계약의'),
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
