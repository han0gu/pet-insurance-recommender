from langchain_core.documents import Document

chunk = Document(
    page_content=("id='225' data-category='paragraph' style='font-size:14px'>약</p><br><p "
 "id='226' data-category='paragraph' style='font-size:14px'>질</p><br><p "
 "id='227' data-category='paragraph' style='font-size:22px'>상해</p><br><p "
 "id='228' data-category='paragraph' style='font-size:14px'>관</p><p id='229'"),
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
