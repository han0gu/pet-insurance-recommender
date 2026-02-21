from langchain_core.documents import Document

chunk = Document(
    page_content=(". 다만,</p><footer id='101' style='font-size:14px'>- 19 -</footer><header "
 "id='102' style='font-size:14px'>회사와 계약자가 합의하여 관할법원을 달리 정할 수 "
 "있습니다.</header><h1 id='103' style='font-size:14px'>제36조(소멸시효)</h1><br><p "
 "id='104' data-category='paragraph' style='font-size:14px'>보험금청구권, 보험료 또는 환급금 "
 '반환청구권은 3년간 행사하지 않으면'),
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
