from langchain_core.documents import Document

chunk = Document(
    page_content=('. 삭제 <2005. 6. 30><br>6. 부처님오신날 (음력 4월 8일)<br>7. 5월 5일 (어린이날)<br>8. 6월 6일 '
 '(현충일)<br>9. 추석 전날, 추석, 추석 다음날 (음력 8월 14일, 15일, 16일)<br>10. 12월 '
 '25일(기독탄신일)<br>10의2. 「공직선거법」제34조에 따른 임기만료에 의한 선거의 선거일<br>11'),
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
