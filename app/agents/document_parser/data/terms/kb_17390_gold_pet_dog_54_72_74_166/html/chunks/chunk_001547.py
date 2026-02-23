from langchain_core.documents import Document

chunk = Document(
    page_content=('부위로 한다.<br>제2천추 이하의 천골 및 미골은 체간골의 장해로 평가한다.<br>2) 척추(등뼈)의 기형장해는 척추체(척추뼈 몸통을 '
 '말하며, 횡돌기 및 극돌<br>기는 제외한다'),
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
