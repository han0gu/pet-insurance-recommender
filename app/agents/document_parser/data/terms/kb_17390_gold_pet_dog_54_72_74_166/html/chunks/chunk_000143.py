from langchain_core.documents import Document

chunk = Document(
    page_content=(". 검진결과 추가검사 또는 치료가 필요하지 않았던 경우</p><br><p id='171' data-category='paragraph' "
 "style='font-size:14px'>2"),
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
