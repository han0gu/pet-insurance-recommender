from langchain_core.documents import Document

chunk = Document(
    page_content=('| 안면과 경부 이외 | 창상봉합술(안면과경부이외,단순봉합,근육,길이2.5cm이상~5.0cm미만) | SB032 |\n'
 '| 안면과 경부 이외 | 창상봉합술(안면과경부이외,단순봉합,근육,길이5.0cm이상~10.0cm미만) | SB039 |\n'
 '| 안면과 경부 이외 | 창상봉합술(안면과경부이외,단순봉합,근육,길이10cm이상, 10cm마다 추가) | SB040 |\n'
 '| 안면과 경부 이외 | 창상봉합술(안면과경부이외,변연절제포함,표재성,길이5.0cm이상~10.0cm미만) | SC029 |'),
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
