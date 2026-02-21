from langchain_core.documents import Document

chunk = Document(
    page_content=("체납된 세금에<br>대하여 가산금징수, 독촉장 발부 및 재산 압류 등의 집행을 하는 것을 말합니다.</p><footer id='77' "
 "style='font-size:14px'>- 17 -</footer><h1 id='78' style='font-size:14px'>제6관 "
 "계약의 해지 및 보험료의 환급 등</h1><h1 id='79' style='font-size:14px'>제30조(계약의 "
 "해지)</h1><br><p id='80' data-category='paragraph' style='font-size:14px'>계약자는 "
 '계약이'),
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
