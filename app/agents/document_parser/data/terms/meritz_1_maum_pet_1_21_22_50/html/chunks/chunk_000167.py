from langchain_core.documents import Document

chunk = Document(
    page_content=("style='font-size:14px'>계약자는 계약이 소멸하기 전에는 언제든지 계약을 해지할 수 있으며, 이 경우 회사가 "
 '환<br>급하여야 할 보험료가 있을 경우에는 제33조(보험료의 환급)에 따른 보험료를 계약자에게<br>지급합니다.</p><h1 '
 "id='81' style='font-size:14px'>제30조의2(위법계약의 해지)</h1><br><p id='82' "
 "data-category='list' style='font-size:14px'>① 계약자는 ｢금융소비자보호에 관한 법률｣ 제47조 및 "
 '관련규정이 정하는 바에 따라'),
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
