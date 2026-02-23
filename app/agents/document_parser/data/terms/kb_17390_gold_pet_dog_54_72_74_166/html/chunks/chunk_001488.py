from langchain_core.documents import Document

chunk = Document(
    page_content=('. 이 경우 시야검사는 공인된 시야검사방법<br>으로 측정하며, 시야장해 평가 시 자동시야검사계(골드만 시야검사)를<br>이용하여 8방향 '
 '시야범위 합계를 정상범위와 비교하여 평가한다.<br>9) ‘눈꺼풀에 뚜렷한 결손을 남긴 때’라 함은 눈꺼풀의 결손으로 눈을 감<br>았을 '
 '때 각막(검은 자위)이 완전히 덮이지 않는 경우를 말한다.<br>10) ‘눈꺼풀에 뚜렷한 운동장해를 남긴 때’ 라 함은 눈을 떴을 때 '
 '동공을<br>1/2 이상 덮거나 또는 눈을 감았을 때 각막을 완전히 덮을 수 없는 경우<br>를 말한다.<br>11) 외상이나'),
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
