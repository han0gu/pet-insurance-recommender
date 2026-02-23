from langchain_core.documents import Document

chunk = Document(
    page_content=(". 도<br>성</p><p id='222' data-category='paragraph' style='font-size:18px'>- "
 "81 -</p><br><p id='223' data-category='paragraph' style='font-size:16px'>KB "
 "금쪽같은 펫보험(강아지)(무배당)(26.01) 81</p><br><p id='224' data-category='paragraph' "
 "style='font-size:14px'>특</p><br><p id='225' data-category='paragraph'"),
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
