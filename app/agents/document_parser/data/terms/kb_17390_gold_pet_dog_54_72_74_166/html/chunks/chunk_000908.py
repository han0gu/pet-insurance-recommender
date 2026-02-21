from langchain_core.documents import Document

chunk = Document(
    page_content=(". 약</p><br><p id='82' data-category='paragraph' style='font-size:16px'>전에는 "
 "언제든지 계약을 해지할 수 있으며, 이 경우</p><br><p id='83' data-category='paragraph' "
 "style='font-size:14px'>및</p><br><p id='84' data-category='paragraph' "
 "style='font-size:14px'>질</p><p id='85' data-category='paragraph'"),
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
