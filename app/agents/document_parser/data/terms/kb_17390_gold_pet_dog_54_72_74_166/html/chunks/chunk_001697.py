from langchain_core.documents import Document

chunk = Document(
    page_content=('동반한 머리의 동상 \u3000조직괴사를 동반한 목의 동상</td><td>T34.0 '
 'T34.1</td></tr><tr><td></td><td></td></tr><tr><td>주) 1'),
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
