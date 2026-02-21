from langchain_core.documents import Document

chunk = Document(
    page_content=("style='font-size:14px'>담보권실행이란 담보권을 설정한 채권자가 채무를 이행하지 않는 채무자에 대 상<br>하여 해당 "
 '담보권을 실행하는 것을 말합니다'),
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
