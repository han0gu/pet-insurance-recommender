from langchain_core.documents import Document

chunk = Document(
    page_content=("style='font-size:20px'>일반상해80%이상후유장해</p><br><p id='190' "
 "data-category='paragraph' style='font-size:14px'>제1조(보험금의 지급사유)<br>회사는 피보험자가 "
 '이 보장의 보험기간 중에 상해로 장해분류표(【별표1】(장해분류<br>표) 참조'),
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
