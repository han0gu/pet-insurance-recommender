from langchain_core.documents import Document

chunk = Document(
    page_content='행위로 인하여 제3조(보험금의 지급사유)의 상해 관련 보험금 지급사<br>유가 발생한 때에는 해당 보험금을 지급하지 않습니다.<br>1',
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
