from langchain_core.documents import Document

chunk = Document(
    page_content=('…" data-coord="top-left:(148,145); bottom-right:(695,326)" '
 "/></figure></td></tr></tbody></table><br><p id='35' "
 "data-category='paragraph' style='font-size:16px'>(180일)</p><br><p id='36' "
 "data-category='paragraph' style='font-size:16px'>(180일)</p><br><p id='37' "
 "data-category='paragraph'"),
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
