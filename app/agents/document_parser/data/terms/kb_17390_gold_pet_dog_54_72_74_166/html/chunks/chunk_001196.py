from langchain_core.documents import Document

chunk = Document(
    page_content=('. 이 경우 대부금의 이자는 공탁금에 붙<br>여지는 것과 같은 이율로 하며, 피보험자는 공탁금(이자를 포함합니다)의 회수청 '
 '반<br>구권을 회사에 양도하여야 합니다'),
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
