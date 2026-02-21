from langchain_core.documents import Document

chunk = Document(
    page_content='청약을 승낙하고 제1회 보험료를 받은 때부터 이 약관이 정한 바에<br>따라 보장을 합니다',
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
