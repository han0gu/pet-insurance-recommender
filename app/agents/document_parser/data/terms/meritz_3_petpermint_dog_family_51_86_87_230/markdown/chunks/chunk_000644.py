from langchain_core.documents import Document

chunk = Document(
    page_content=('- 관절(주관절), 손목관절(완관절)을 말한다.\n'
 '- 5) “한팔의 손목이상을 잃었을 때”라 함은 손목관절\n'
 '- (완관절)부터(손목관절 포함) 심장에 가까운 쪽에서\n'
 '- 절단된 때를 말하며, 팔꿈치관절(주관절) 상부에서\n'
 '- 절단된 경우도 포함한다.\n'
 '- 6) 팔의 관절기능 장해 평가는 팔의 3대관절의 관절운동\n'
 '- 범위 제한 등으로 평가한다.\n'
 '- 가) 각 관절의 운동범위 측정은 장해평가시점의 ｢산\n'
 '- 업재해보상보험법 시행규칙｣ 제47조 제1항 및 제\n'
 '- 3항의 정상인의 신체 각 관절에 대한 평균 운동'),
    metadata={'source_doc': {'total_pages': 180},
 'doc': {'doc_type': 'terms',
         'file_name': 'meritz_3_petpermint_dog_family_51_86_87_230.pdf',
         'insurer_code': 'meritz',
         'product_code': '3',
         'product_name': '무배당 펫퍼민트 Puppy&Family보험 다이렉트2601',
         'total_pages': 1,
         'page': 1},
 'term_type': 'unknown'},
)
