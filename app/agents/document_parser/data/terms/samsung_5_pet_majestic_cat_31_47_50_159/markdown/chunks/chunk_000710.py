from langchain_core.documents import Document

chunk = Document(
    page_content=('| 53 | 관절증 및 류마티스 관절염 | M18 | 제1수근중수관절의 관절증 |\n'
 '| 53 | 관절증 및 류마티스 관절염 | M19 | 기타 관절증 |\n'
 '| 54 | 척추질환 | M47 | 척추증 |\n'
 '| 54 | 척추질환 | M48.0 | 척추협착 |\n'
 '| 54 | 척추질환 | M50 | 경추간판장애 |\n'
 '| 54 | 척추질환 | M51 | 기타 추간판장애 |\n'
 '| 54 | 척추질환 | M54 | 등통증 |\n'
 '| 55 | 골반염 | N73 | 기타 여성골반염증질환 |'),
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
