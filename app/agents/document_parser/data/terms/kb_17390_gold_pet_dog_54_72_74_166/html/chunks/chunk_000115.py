from langchain_core.documents import Document

chunk = Document(
    page_content='. 그러지 않을 경우 보험사고가 발생한 경우에도 보험금 지급이 제한될 수 있습니다',
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
