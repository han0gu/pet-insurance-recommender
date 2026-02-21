from langchain_core.documents import Document

chunk = Document(
    page_content=('계약은 대한민국 법에 따라 규율되고 해석되며, 약관에서 정하지 않은 사항은 「금융소<br>비자보호에 관한 법률」, 상법, 민법 등 관계 '
 "법령을 따릅니다.</p><h1 id='8' style='font-size:14px'>제42조(예금보험에 의한 "
 "지급보장)</h1><br><p id='9' data-category='paragraph' style='font-size:14px'>회사가 "
 '파산 등으로 인하여 보험금 등을 지급하지 못할 경우에는 예금자보호법에서 정하는<br>바에 따라 그 지급을 보장합니다.</p><h1 '
 "id='10'"),
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
