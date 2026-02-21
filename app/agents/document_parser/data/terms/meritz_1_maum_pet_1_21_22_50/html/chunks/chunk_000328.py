from langchain_core.documents import Document

chunk = Document(
    page_content=("정산방법)</h1><br><p id='84' data-category='paragraph' "
 "style='font-size:14px'>보험료는 보험의 목적의 정보의 변경을 기초로 하여 다음과 같이 정산합니다.</p><br><p "
 "id='85' data-category='list' style='font-size:14px'>1"),
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
