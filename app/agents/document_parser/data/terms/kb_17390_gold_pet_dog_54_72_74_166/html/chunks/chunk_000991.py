from langchain_core.documents import Document

chunk = Document(
    page_content=('국내에서 수의사에게 "반려동물주요치료"를 받은 경우에는 치료구분별로 각<br>각의 지급방식에 따라 당일 피보험자가 부담한 반려동물의 '
 '치료에 사용된 비용(각<br>종 할인 및 감면, 사후환급금액 등을 제외한 실수납액을 의미합니다'),
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
