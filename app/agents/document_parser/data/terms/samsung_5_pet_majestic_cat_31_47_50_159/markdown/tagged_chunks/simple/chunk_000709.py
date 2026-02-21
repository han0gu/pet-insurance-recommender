from langchain_core.documents import Document

chunk = Document(
    page_content=('| 53 | 관절증 및 류마티스 관절염 | M06 | 기타 류마티스관절염 |\n'
 '| 53 | 관절증 및 류마티스 관절염 | M08 | 연소성 관절염 |\n'
 '| 53 | 관절증 및 류마티스 관절염 | M13 | 기타 관절염 |\n'
 '| 53 | 관절증 및 류마티스 관절염 | M15 | 다발관절증 |\n'
 '| 53 | 관절증 및 류마티스 관절염 | M16 | 고관절증 |\n'
 '| 53 | 관절증 및 류마티스 관절염 | M17 | 무릎관절증 |\n'
 '| 53 | 관절증 및 류마티스 관절염 | M18 | 제1수근중수관절의 관절증 |'),
    metadata={'source_doc': {'total_pages': 129},
 'doc': {'doc_type': 'terms',
         'file_name': 'samsung_5_pet_majestic_cat_31_47_50_159.pdf',
         'insurer_code': 'samsung',
         'product_code': '5',
         'product_name': '(장기)무배당 삼성화재 펫보험 의기냥냥',
         'total_pages': 1,
         'page': 1},
 'term_type': 'unknown',
 'clause': {'clause_type': 'other', 'risk_domains': ['joint']},
 'indexing': {'chunk_id': 'chunk_000709',
              'chunk_char_len': 280,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'simple',
              'tag_method': 'rule',
              'tag_confidence': 0.25}},
)
