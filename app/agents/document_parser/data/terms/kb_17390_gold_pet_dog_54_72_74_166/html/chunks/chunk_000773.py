from langchain_core.documents import Document

chunk = Document(
    page_content=(". 다만, 재외공관의 공휴일은 우리나라</p><br><p id='129' data-category='paragraph' "
 "style='font-size:16px'>의</p><br><p id='130' data-category='list' "
 "style='font-size:16px'>국경일 중 공휴일과 주재국의 공휴일로 한다.<br>1. 일요일<br>2. 국경일 중 3 ‧ "
 '1절, 광복절, 개천절 및 한글날<br>3. 1월 1일<br>4. 설날 전날, 설날, 설날 다음날 (음력 12월 말일, 1월 1일, '
 '2일)<br>5. 삭제 <2005'),
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
