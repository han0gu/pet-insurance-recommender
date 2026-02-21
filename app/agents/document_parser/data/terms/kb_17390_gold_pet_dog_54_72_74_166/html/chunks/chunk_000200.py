from langchain_core.documents import Document

chunk = Document(
    page_content=("수 있으며,<br>계약내용 변경시점 이후 잔여 보험기간의 보장을 위한 재원인 계약자적립액 등의</p><br><p id='11' "
 "data-category='paragraph' style='font-size:18px'>- 64 -</p><p id='12' "
 "data-category='paragraph' style='font-size:16px'>차이로 계약자가 추가로 납입하여야 할 (또는 "
 "반환받을) 금액이 발생할 수 있습니<br>다.</p><br><h1 id='13' style='font-size:16px'>\uf000 "
 '제1항에 따라 보험료 등의'),
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
