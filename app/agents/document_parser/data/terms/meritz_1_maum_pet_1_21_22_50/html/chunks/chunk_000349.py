from langchain_core.documents import Document

chunk = Document(
    page_content=("id='20' data-category='paragraph' style='font-size:14px'>① 계약자는 다음의 서류를 제출하고 "
 '지정대리청구인을 변경 지정할 수 있습니다'),
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
