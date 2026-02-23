from langchain_core.documents import Document

chunk = Document(
    page_content=('보험료 또는 환급금 반환청구권은 3년간 행사하지 않으면 소멸시효(소멸시<br>효는 해당 청구권을 행사할 수 있는 때로부터 진행합니다.)가 '
 "완성됩니다.</p><br><h1 id='105' style='font-size:14px'>【소멸시효】</h1><br><p id='106' "
 "data-category='paragraph' style='font-size:14px'>주어진 권리를 행사하지 않을 때 그 권리가 "
 '없어지게 되는 기간으로 보험금 지급사유가<br>발생한 후 3년간 보험금을 청구하지 않는 경우 보험금을 지급받지 못할 수'),
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
