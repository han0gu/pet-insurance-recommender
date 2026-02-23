from langchain_core.documents import Document

chunk = Document(
    page_content=('관계)<br>\uf000 회사는 이 특별약관에 따라 보상하여야 하는 금액이 의무보험에서 보상하는 금액<br>을 초과할 때에 한하여 그 '
 '초과액만을 보상합니다'),
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
