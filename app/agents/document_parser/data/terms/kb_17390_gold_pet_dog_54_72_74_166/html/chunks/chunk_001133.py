from langchain_core.documents import Document

chunk = Document(
    page_content=(". 다</p><br><p id='145' data-category='paragraph' style='font-size:14px'>만, 이 "
 "특별약관에서는 반려동물(강아지) 일반조항 제22조(재가입)은 제외합니<br>약</p><br><p id='146' "
 "data-category='paragraph' style='font-size:14px'>성</p><br><p id='147' "
 "data-category='paragraph' style='font-size:14px'>특</p><p id='148'"),
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
