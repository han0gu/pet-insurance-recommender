from langchain_core.documents import Document

chunk = Document(
    page_content=('따라 보험금 지급여부가 판단된 경우, 이후 수가코드가 변경되더라도 이 약관에서 보장하\n'
 '는 의료행위 해당 여부를 다시 판단하지 않습니다.| 분류항목 | 분류항목 | 수가코드 |\n'
 '| --- | --- | --- |\n'
 '| 안면 또는 경부 | 창상봉합술(안면또는경부,단순봉합,표재성,길이 1.5cm 미만) | S0021 |\n'
 '| 안면 또는 경부 | 창상봉합술(안면또는경부,단순봉합,표재성,길이 1.5cm 이상~3.0cm 미만) | S0022 |'),
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
