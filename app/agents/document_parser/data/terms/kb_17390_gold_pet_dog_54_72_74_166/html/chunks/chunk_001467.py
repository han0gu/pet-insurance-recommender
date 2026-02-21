from langchain_core.documents import Document

chunk = Document(
    page_content=('. 장해의 정의<br>1) ‘장해’라 함은 상해 또는 질병에 대하여 치유된 후 신체에 남아 있는 영구<br>적인 정신 또는 육체의 '
 '훼손상태 및 기능상실 상태를 말한다'),
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
