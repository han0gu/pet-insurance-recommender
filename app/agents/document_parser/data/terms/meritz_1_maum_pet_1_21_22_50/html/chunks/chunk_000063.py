from langchain_core.documents import Document

chunk = Document(
    page_content=("style='font-size:14px'>【가지급보험금】</p><br><p id='70' data-category='paragraph' "
 "style='font-size:14px'>보험금이 지급기한 내에 지급되지 못할 것으로 판단되는 경우 회사가 예상되는 보험<br>금의 "
 '일부를 먼저 지급하는 제도로 피보험자가 필요로 하는 비용을 보전해 주기 위<br>해 회사가 먼저 지급하는 임시 교부금을 '
 "말합니다.</p><br><p id='71' data-category='list' style='font-size:14px'>④ 회사는 "
 '제1항의 규정에'),
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
