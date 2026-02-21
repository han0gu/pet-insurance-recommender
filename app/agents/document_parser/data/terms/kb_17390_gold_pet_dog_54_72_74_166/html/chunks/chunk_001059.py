from langchain_core.documents import Document

chunk = Document(
    page_content=("style='font-size:16px'>이 법에서 사용하는 용어의 뜻은 다음과 같다.</h1><br><p id='31' "
 "data-category='list' style='font-size:14px'>1"),
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
