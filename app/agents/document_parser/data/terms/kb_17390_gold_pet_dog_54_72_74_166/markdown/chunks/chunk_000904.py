from langchain_core.documents import Document

chunk = Document(
    page_content=('- 무릎관절(슬관절)의 동요성 등으로 평가한다.\n'
 '- 가) 각 관절의 운동범위 측정은 장해평가시점의 ｢산업재해보상보험법\n'
 '- 시행규칙｣ 제47조 제1항 및 제3항의 정상인의 신체 각 관절에 대한\n'
 '- 평균 운동가능영역을 기준으로 정상각도 및 측정방법 등을 따른다.\n'
 '- 나) 관절기능장해가 신경손상으로 인한 경우에는 운동범위 측정이 아닌\n'
 '근력 및 근전도 검사를 기준으로 평가한다.\n'
 '7) 관절 하나의 기능을 완전히 잃었을 때’라 함은 아래의 경우 중 하나에- 해당하는 때를 말한다.\n'
 '- 가) 완전 강직(관절굳음)'),
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
