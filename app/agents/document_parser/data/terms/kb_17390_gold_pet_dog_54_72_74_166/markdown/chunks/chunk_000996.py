from langchain_core.documents import Document

chunk = Document(
    page_content=('| 창상봉합술Ⅱ (급여) (안면/경부 | 대상이 되는 항목 | 수가코드 |\n'
 '| 창상봉합술Ⅱ (급여) (안면/경부 | 창상봉합술 |  |\n'
 '| 창상봉합술Ⅱ (급여) (안면/경부 | 나. 안면 또는 경부 이외 |  |\n'
 '| 창상봉합술Ⅱ (급여) (안면/경부 | (1) 단순봉합 |  |\n'
 '| 창상봉합술Ⅱ (급여) (안면/경부 | (가) 표재성인 것 3) 길이 5.0cm 이상 ~ 10.0cm 미만 | SB029 |'),
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
