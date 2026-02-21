from langchain_core.documents import Document

chunk = Document(
    page_content=("style='font-size:14px'>약</p><br><p id='43' data-category='paragraph' "
 "style='font-size:14px'>관</p><br><p id='44' data-category='paragraph' "
 "style='font-size:16px'>제4조(보험금의 청구)</p><br><p id='45' "
 "data-category='paragraph' style='font-size:16px'>\uf000</p><br><p id='46' "
 "data-category='list'"),
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
