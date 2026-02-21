from langchain_core.documents import Document

chunk = Document(
    page_content=('| 안면 또는 경부 | 창상봉합술(안면또는경부,변연절제포함,표재성,길이10cm이상, 5cm마다 추가) | SA030 |\n'
 '| 안면 또는 경부 | 창상봉합술(안면또는경부,변연절제포함,근육,길이1.5cm미만) | SA031 |\n'
 '| 안면 또는 경부 | 창상봉합술(안면또는경부,변연절제포함,근육,길이1.5cm이상~3.0cm미만) | SA032 |\n'
 '| 안면 또는 경부 | 창상봉합술(안면또는경부,변연절제포함,근육,길이3.0cm이상~5.0cm미만) | SA037 |'),
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
