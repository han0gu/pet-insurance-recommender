from langchain_core.documents import Document

chunk = Document(
    page_content=('. 수의사의 치료상의 과오로 생긴 상해 또는 질병, 수의사 자격이 없는 자의 치료행위<br>로 인한 비용 및 그로 인하여 가중된 '
 "비용</p><br><p id='45' data-category='list' style='font-size:14px'>10. 국가 및 "
 '지방자치단체의 명령 또는 법률에 의한 살처분 또는 이와 유사한 사태<br>11'),
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
