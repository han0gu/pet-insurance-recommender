from langchain_core.documents import Document

chunk = Document(
    page_content=("id='68' data-category='paragraph' style='font-size:14px'>보험료 납입을 연체하여 계약이 "
 '해지되고 계약자가 해약환급금을 받지 않은 경우<br>회사가 정하는 소정의 절차에 따라 해지된 계약을 다시 되살리는 것을 '
 "말합니다.</p><h1 id='69' style='font-size:14px'>제29조(강제집행 등의 절차에 따라 해지된 계약의 "
 "특별부활(효력회복))</h1><br><p id='70' data-category='list' style='font-size:14px'>① "
 '타인을 위한'),
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
