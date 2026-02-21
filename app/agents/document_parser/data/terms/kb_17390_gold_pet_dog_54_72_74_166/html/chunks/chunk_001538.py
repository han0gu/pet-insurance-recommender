from langchain_core.documents import Document

chunk = Document(
    page_content=("id='20' data-category='paragraph' style='font-size:14px'>손바닥 크기 이상의 추상(추한 "
 "모습)</p><p id='21' data-category='list'></p><br><p id='22' "
 "data-category='paragraph' style='font-size:14px'>라.</p><br><h1 id='23' "
 "style='font-size:14px'>약간의</h1><br><p id='24' "
 "data-category='list'></p><br><p id='25'"),
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
