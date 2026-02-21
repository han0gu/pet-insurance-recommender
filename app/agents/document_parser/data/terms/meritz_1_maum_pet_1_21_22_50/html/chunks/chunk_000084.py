from langchain_core.documents import Document

chunk = Document(
    page_content=('후 보험의 목적에 아래와 같은 사실이 생긴 경우에는 계약자나 피보험자는<br>지체없이 서면으로 회사에 알리고 보험증권에 확인을 받아야 '
 "합니다.</p><br><p id='107' data-category='list' style='font-size:14px'>1"),
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
