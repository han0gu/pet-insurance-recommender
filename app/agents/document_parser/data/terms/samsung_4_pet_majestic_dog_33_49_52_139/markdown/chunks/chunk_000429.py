from langchain_core.documents import Document

chunk = Document(
    page_content=('| 보장구분 | 지급기준 | 지급기준 |\n'
 '| --- | --- | --- |\n'
 '| 보장구분 | 안면 또는 경부 | 안면과 경부 이외 |\n'
 '| 창상봉합술(급여)(B) | 3cm이상 또는 근육에 달하는 것 | 5cm 이상 또는 근육에 달하는 것 |\n'
 '| 안면부 창상봉합술 (급여)(C) | 3cm이상 또는 근육에 달하는 것 | 미보장 |\n'
 '| 안면부 창상봉합술 (단순봉합 제외,급여)(D) | 3cm이상 또는 근육에 달하는 것 (단순봉합 제외) | 미보장 |'),
    metadata={'source_doc': {'total_pages': 107},
 'doc': {'doc_type': 'terms',
         'file_name': 'samsung_4_pet_majestic_dog_33_49_52_139.pdf',
         'insurer_code': 'samsung',
         'product_code': '4',
         'product_name': '(장기)무배당 삼성화재 펫보험 위풍댕댕',
         'total_pages': 1,
         'page': 1},
 'term_type': 'unknown'},
)
