from langchain_core.documents import Document

chunk = Document(
    page_content=(". 보험종목</p><br><p id='29' data-category='paragraph' "
 "style='font-size:14px'>및</p><br><p id='30' data-category='paragraph' "
 "style='font-size:14px'>질</p><p id='31' data-category='list' "
 "style='font-size:16px'>2. 보험기간<br>3. 보험료 납입방법 및 납입기간<br>4. 계약자, 피보험자<br>5"),
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
