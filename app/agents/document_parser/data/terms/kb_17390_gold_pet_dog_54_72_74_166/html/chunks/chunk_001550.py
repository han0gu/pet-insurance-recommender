from langchain_core.documents import Document

chunk = Document(
    page_content=('. 척추체(척추뼈 몸통)의 압박률은 인<br>접 상․하부[인접 상․하부 척추체(척추뼈 몸통)에 진구성 골절이 있거<br>나, 다발성 '
 '척추골절이 있는 경우에는 골절된 척추와 가장 인접한<br>상 ․하부] 정상 척추체(척추뼈 몸통)의 전방 높이의 평균에 대한 골절<br>된 '
 '척추체(척추뼈 몸통) 전방 높이의 감소비를 압박률로 정한다.<br>다) 척추(등뼈)의 기형장해는 ｢산업재해보상보험법 시행규칙｣상 '
 '경추<br>부, 흉추부, 요추부로 구분하여 각각을 하나의 운동단위로 보며, 하<br>나의 운동단위 내에서 여러 개의 척추체(척추뼈 '
 '몸통)에'),
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
