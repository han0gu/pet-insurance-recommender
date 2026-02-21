from langchain_core.documents import Document

chunk = Document(
    page_content=('. 다만, 회사가 나이의 착오를 발견하였을 때 이미 계약나이에 도달한 경<br>우에는 유효한 계약으로 보나, 제2호의 만15세 미만자에 '
 '관한 예외가 인정되는<br>것은 아닙니다.<br>용 어 풀 이<br>∙ 심신상실자<br>심신상실자(心神喪失者)라 함은 의식은 있으나 '
 '장애의 정도가 심하여 자신의<br>행위 결과를 합리적으로 판단할 능력을 갖지 못한 사람을 말합니다.<br>∙ '
 '심신박약자<br>심신박약자(心神薄弱者)라 함은 심신상실의 상태까지는 이르지 않았으나, 마음<br>이나 정신의 장애로 인하여 사물을 변별할 '
 '능력이나 의사를 결정할'),
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
