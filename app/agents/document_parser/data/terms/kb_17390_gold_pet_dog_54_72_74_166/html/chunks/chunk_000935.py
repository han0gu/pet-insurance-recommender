from langchain_core.documents import Document

chunk = Document(
    page_content=('. 이하<br>의료비라 합니다)을 제2항에 따라 이 특별약관의 보험가입금액을 한도로 보험수익<br>상<br>자에게 '
 '반려동물의료비보험금(이하 의료비보험금이라 합니다)으로 보상하여 드립<br>해<br>니다'),
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
