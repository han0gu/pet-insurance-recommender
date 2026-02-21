from langchain_core.documents import Document

chunk = Document(
    page_content=('1969) 손가락의 관절기능장해 평가는 손가락 관절의 관절운\n'
 '동범위 제한 등으로 평가한다. 각 관절의 운동범위\n'
 '측정은 장해평가시점의 ｢산업재해보상보험법 시행규\n'
 '칙｣ 제47조 제1항 및 제3항의 정상인의 신체 각 관절\n'
 '에 대한 평균 운동가능영역을 기준으로 정상각도 및\n'
 '측정방법 등을 따른다.![image](/image/placeholder)\n'
 '# < 손가락 ># 11. 발가락의 장해# 가. 장해의 분류| 장해의 분류 | 지급률 |\n'
 '| --- | --- |\n'
 '| 1) 한발의 리스프랑관절 이상을 잃었을 때 | 40 |'),
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
