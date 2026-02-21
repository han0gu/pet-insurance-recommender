from langchain_core.documents import Document

chunk = Document(
    page_content=('. 다만, 조절력의 감소를 무시할 수 있는<br>50세 이상(장해진단시 연령 기준)의 경우에는 제외한다.<br>8) ‘뚜렷한 시야 '
 '장해’라 함은 한 눈의 시야 범위가 정상시야 범위의 60%<br>이하로 제한된 경우를 말한다'),
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
