from langchain_core.documents import Document

chunk = Document(
    page_content=("style='font-size:14px'>예 시</p><br><p id='45' data-category='paragraph' "
 "style='font-size:14px'>∙ 계약해당일 계산<br>최초계약일과 동일한 월, 일을 말합니다"),
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
