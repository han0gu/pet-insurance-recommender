from langchain_core.documents import Document

chunk = Document(
    page_content='. 현재의 직업 또는 직무가 변경된 경우<br>나. 직업이 없는 자가 취직한 경우<br>다',
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
