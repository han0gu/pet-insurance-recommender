from langchain_core.documents import Document

chunk = Document(
    page_content=('가까운 쪽에서 절단된 때를 말하며, 무릎관절(슬관<br>절)의 상부에서 절단된 경우도 포함한다.</header><br><p '
 "id='108' data-category='list' style='font-size:16px'>6) 다리의 관절기능장해 평가는 다리의 "
 '3대 관절의 관절운동범위 제한 및<br>무릎관절(슬관절)의 동요성 등으로 평가한다.<br>가) 각 관절의 운동범위 측정은 장해평가시점의 '
 '｢산업재해보상보험법<br>시행규칙｣ 제47조 제1항 및 제3항의 정상인의 신체 각 관절에 대한<br>평균 운동가능영역을 기준으로 정상각도 '
 '및'),
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
