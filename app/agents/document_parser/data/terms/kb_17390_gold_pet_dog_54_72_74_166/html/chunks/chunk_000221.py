from langchain_core.documents import Document

chunk = Document(
    page_content='. 또한, 회사가 청약과 함께 제1회 보험료를 받은 후 승낙한<br>경우에도 제1회 보험료를 받은 때부터 보장이 개시됩니다',
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
