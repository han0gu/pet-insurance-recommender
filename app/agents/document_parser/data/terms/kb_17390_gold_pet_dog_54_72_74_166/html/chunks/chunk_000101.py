from langchain_core.documents import Document

chunk = Document(
    page_content=("id='127' data-category='paragraph' style='font-size:16px'>\uf000</p><br><p "
 "id='128' data-category='paragraph' style='font-size:16px'>변경 등)에 따라 계약내용을 "
 "변경할 수 있습니다.</p><p id='129' data-category='paragraph' "
 "style='font-size:18px'>- 59 -</p><br><p id='130' data-category='paragraph' "
 "style='font-size:16px'>KB"),
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
