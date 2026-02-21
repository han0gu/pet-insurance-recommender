from langchain_core.documents import Document

chunk = Document(
    page_content=('| 53 | 관절증 및 류마티스 관절염 | M06 | 기타 류마티스관절염 |\n'
 '| 53 | 관절증 및 류마티스 관절염 | M08 | 연소성 관절염 |\n'
 '| 53 | 관절증 및 류마티스 관절염 | M13 | 기타 관절염 |\n'
 '| 53 | 관절증 및 류마티스 관절염 | M15 | 다발관절증 |\n'
 '| 53 | 관절증 및 류마티스 관절염 | M16 | 고관절증 |\n'
 '| 53 | 관절증 및 류마티스 관절염 | M17 | 무릎관절증 |\n'
 '| 53 | 관절증 및 류마티스 관절염 | M18 | 제1수근중수관절의 관절증 |'),
    metadata={'source_doc': {'total_pages': 78},
 'doc': {'doc_type': 'terms',
         'file_name': 'samsung_3_direct_good_pet_28_42_45_105.pdf',
         'insurer_code': 'samsung',
         'product_code': '3',
         'product_name': '(장기)무배당 삼성화재 다이렉트 '
                         '착한펫보험(강아지)',
         'total_pages': 1,
         'page': 1},
 'term_type': 'unknown',
 'clause': {'clause_type': 'other', 'risk_domains': ['joint']},
 'indexing': {'chunk_id': 'chunk_000577',
              'chunk_char_len': 280,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'simple',
              'tag_method': 'rule',
              'tag_confidence': 0.25}},
)
