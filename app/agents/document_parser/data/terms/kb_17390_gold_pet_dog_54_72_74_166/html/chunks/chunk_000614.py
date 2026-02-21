from langchain_core.documents import Document

chunk = Document(
    page_content=(". 특<br>약</p><br><p id='127' data-category='paragraph' "
 "style='font-size:16px'>특수 다수인을 위하여 의</p><br><p id='128' "
 "data-category='paragraph' style='font-size:14px'>제</p><p id='129' "
 "data-category='paragraph' style='font-size:16px'>KB 금쪽같은 "
 "펫보험(강아지)(무배당)(26.01) 87</p><br><p id='130'"),
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
