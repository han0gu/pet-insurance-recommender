from langchain_core.documents import Document

chunk = Document(
    page_content=('| 창상봉합술Ⅰ (급여) (안면/경부) | 가. 안면 또는 경부 | 보 |\n'
 '| 창상봉합술Ⅰ (급여) (안면/경부) | (1) 단순봉합 | 통약 |\n'
 '| 창상봉합술Ⅰ (급여) (안면/경부) | (가) 표재성인 것 |  |\n'
 '| 창상봉합술Ⅰ (급여) (안면/경부) | 1) 길이 1.5cm 미만 S0021 | 관 |\n'
 '| 창상봉합술Ⅰ (급여) (안면/경부) | 2) 길이 1.5cm 이상 ~ 3.0cm 미만 S0022 |  |\n'
 '| 창상봉합술Ⅰ (급여) (안면/경부) | (2) 변연절제를 포함 (가) 표재성인 것 |  |'),
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
