from langchain_core.documents import Document

chunk = Document(
    page_content=(". 반<br>려동</p><br><p id='134' data-category='list'></p><br><p id='135' "
 "data-category='paragraph' style='font-size:14px'>질</p><p id='136' "
 "data-category='paragraph' style='font-size:16px'>제8조(보험료의 납입을 연체하여 해지된 계약의 "
 "부활(효력회복))</p><br><p id='137' data-category='paragraph' "
 "style='font-size:14px'>병</p><p"),
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
