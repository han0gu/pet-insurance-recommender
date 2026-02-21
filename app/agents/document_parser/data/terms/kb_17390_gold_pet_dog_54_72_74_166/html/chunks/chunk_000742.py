from langchain_core.documents import Document

chunk = Document(
    page_content=("관련 특별약관</p><p id='79' data-category='paragraph' style='font-size:14px'>- 97 "
 "-</p><h1 id='80' style='font-size:20px'>제3장 상해 및 질병 관련 특별약관</h1><h1 id='81' "
 "style='font-size:18px'>1"),
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
