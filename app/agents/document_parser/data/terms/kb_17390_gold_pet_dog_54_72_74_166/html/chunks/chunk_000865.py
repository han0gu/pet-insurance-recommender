from langchain_core.documents import Document

chunk = Document(
    page_content=("105 -</p><br><p id='38' data-category='paragraph' "
 "style='font-size:14px'>물</p><p id='39' data-category='paragraph' "
 "style='font-size:14px'>\uf000 회사는 제1항에 따라 계약자를 변경한 경우, 변경된 계약자에게 보험증권 및 "
 "약</p><br><p id='40' data-category='paragraph' style='font-size:14px'>관을 교부하고 "
 '변경된 계약자가 요청하는 경우 약관의 중요한 내용을'),
    metadata={'source_doc': {'total_pages': 113},
 'doc': {'doc_type': 'terms',
         'file_name': 'kb_17390_gold_pet_dog_54_72_74_166.pdf',
         'insurer_code': 'kb',
         'product_code': '17390',
         'product_name': '[일반보험] KB반려행복펫보험',
         'total_pages': 1,
         'page': 1},
 'term_type': 'unknown'},
)
