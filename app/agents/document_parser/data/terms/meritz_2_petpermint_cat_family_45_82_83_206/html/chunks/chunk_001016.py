from langchain_core.documents import Document

chunk = Document(
    page_content=('각 관절의 운동범위 측정은 장해평가시점의 ｢산<br>업재해보상보험법 시행규칙｣ 제47조 제1항 및 제<br>3항의 정상인의 신체 각 '
 '관절에 대한 평균 운동<br>가능영역을 기준으로 정상각도 및 측정방법 등을<br>따른다.<br>나) 관절기능장해를 표시할 경우 장해부위의 '
 '장해각<br>도와 정상부위의 측정치를 동시에 판단하여 장해<br>상태를 명확히 한다'),
    metadata={'source_doc': {'total_pages': 160},
 'doc': {'doc_type': 'terms',
         'file_name': 'meritz_2_petpermint_cat_family_45_82_83_206.pdf',
         'insurer_code': 'meritz',
         'product_code': '2',
         'product_name': '무배당 펫퍼민트 Cat&Family보험 다이렉트2601',
         'total_pages': 1,
         'page': 1},
 'term_type': 'unknown'},
)
