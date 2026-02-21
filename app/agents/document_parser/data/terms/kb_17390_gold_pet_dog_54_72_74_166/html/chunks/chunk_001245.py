from langchain_core.documents import Document

chunk = Document(
    page_content=(". 특<br>약</p><br><p id='56' data-category='paragraph' "
 "style='font-size:14px'>제</p><br><p id='57' data-category='paragraph' "
 "style='font-size:14px'>병</p><p id='58' data-category='paragraph' "
 "style='font-size:16px'>KB 금쪽같은 펫보험(강아지)(무배당)(26.01) 125</p><br><p id='59' "
 "data-category='paragraph'"),
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
