from langchain_core.documents import Document

chunk = Document(
    page_content=("해당하는 약물이 명시된 수<br>질<br>의사처방전</p><br><p id='27' data-category='list' "
 "style='font-size:14px'>병<br>5"),
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
