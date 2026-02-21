from langchain_core.documents import Document

chunk = Document(
    page_content=("한 다리가 1cm 이상 짧아지거나 길어진 때</td><td>5</td></tr></tbody></table><br><p id='100' "
 "data-category='list'></p><br><h1 id='101' style='font-size:16px'>나"),
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
