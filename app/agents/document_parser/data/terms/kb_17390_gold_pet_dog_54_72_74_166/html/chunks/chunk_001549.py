from langchain_core.documents import Document

chunk = Document(
    page_content='평가한다.<br>나) 척추(등뼈)의 기형장해는 척추체(척추뼈 몸통)의 압박률, 골절의<br>부위 등을 기준으로 판정한다',
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
