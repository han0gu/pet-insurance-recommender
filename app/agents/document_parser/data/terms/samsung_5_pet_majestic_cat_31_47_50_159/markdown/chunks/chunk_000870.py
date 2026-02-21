from langchain_core.documents import Document

chunk = Document(
    page_content=('치점수」 개정에 따라 수가코드가 변경된 경우에는 개정된 기준을 적용합니다. 다만, 의\n'
 '료행위당시의 「건강보험 행위 급여·비급여 목록 및 급여 상대가치점수」에 따라 보험금\n'
 '지급여부가 판단된 경우, 이후 수가코드가 변경되더라도 이 약관에서 보장하는 의료행위\n'
 '해당 여부를 다시 판단하지 않습니다.| 분류항목 | 분류항목 | 수가코드 |\n'
 '| --- | --- | --- |\n'
 '| 안면 또는 경부 | 창상봉합술(안면또는경부,단순봉합,표재성,길이 3.0cm 이상~5.0cm 미만) | S0027 |'),
    metadata={'source_doc': {'total_pages': 129},
 'doc': {'doc_type': 'terms',
         'file_name': 'samsung_5_pet_majestic_cat_31_47_50_159.pdf',
         'insurer_code': 'samsung',
         'product_code': '5',
         'product_name': '(장기)무배당 삼성화재 펫보험 의기냥냥',
         'total_pages': 1,
         'page': 1},
 'term_type': 'unknown'},
)
