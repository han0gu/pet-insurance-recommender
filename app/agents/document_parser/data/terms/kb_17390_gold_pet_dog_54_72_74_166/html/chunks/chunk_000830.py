from langchain_core.documents import Document

chunk = Document(
    page_content='↓\n계약자, 피보험자의 계약변경사항 확인 후 청약\n↓\n계약변경사항 인수 심사\n↓\n정산금액 처리(환급 또는 추가납입)\n↓',
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
