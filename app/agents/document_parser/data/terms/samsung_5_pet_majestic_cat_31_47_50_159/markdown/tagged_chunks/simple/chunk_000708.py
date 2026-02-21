from langchain_core.documents import Document

chunk = Document(
    page_content=('| 51 | 담석증 | K80 | 담석증 |\n'
 '| 52 | 요로결석증 | N20 | 신장 및 요관의 결석 |\n'
 '| 52 | 요로결석증 | N21 | 하부요로의 결석 |\n'
 '| 52 | 요로결석증 | N22 | 달리 분류된 질환에서의 요로의 결석 |\n'
 '| 52 | 요로결석증 | N23 | 상세불명의 신장 급통증 |\n'
 '| 53 | 관절증 및 류마티스 관절염 | M05 | 혈청검사양성 류마티스관절염 |\n'
 '| 53 | 관절증 및 류마티스 관절염 | M06 | 기타 류마티스관절염 |'),
    metadata={'source_doc': {'total_pages': 129},
 'doc': {'doc_type': 'terms',
         'file_name': 'samsung_5_pet_majestic_cat_31_47_50_159.pdf',
         'insurer_code': 'samsung',
         'product_code': '5',
         'product_name': '(장기)무배당 삼성화재 펫보험 의기냥냥',
         'total_pages': 1,
         'page': 1},
 'term_type': 'unknown',
 'clause': {'clause_type': 'other',
            'risk_domains': ['digestive', 'joint', 'urinary']},
 'indexing': {'chunk_id': 'chunk_000708',
              'chunk_char_len': 260,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'simple',
              'tag_method': 'rule',
              'tag_confidence': 0.25}},
)
