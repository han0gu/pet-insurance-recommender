from langchain_core.documents import Document

chunk = Document(
    page_content=('N22 | 달리 분류된 질환에서의 요로의 결석\n'
 'N23 | 상세불명의 신장 급통증\n'
 '53 | 관절증 및 류마티스 관절염 | M05 | 혈청검사양성 류마티스관절염\n'
 'M06 | 기타 류마티스관절염\n'
 'M08 | 연소성 관절염\n'
 'M13 | 기타 관절염\n'
 'M15 | 다발관절증\n'
 'M16 | 고관절증\n'
 'M17 | 무릎관절증\n'
 'M18 | 제1수근중수관절의 관절증\n'
 'M19 | 기타 관절증\n'
 '54 | 척추질환 | M47 | 척추증\n'
 'M48.0 | 척추협착\n'
 'M50 | 경추간판장애\n'
 'M51 | 기타 추간판장애\n'
 'M54 | 등통증'),
    metadata={'source_doc': {'total_pages': 129},
 'doc': {'doc_type': 'terms',
         'file_name': 'samsung_5_pet_majestic_cat_31_47_50_159.pdf',
         'insurer_code': 'samsung',
         'product_code': '5',
         'product_name': '(장기)무배당 삼성화재 펫보험 의기냥냥',
         'total_pages': 107,
         'page': 130},
 'term_type': 'special',
 'clause': {'clause_type': 'other',
            'risk_domains': ['urinary', 'joint', 'digestive', 'other']},
 'indexing': {'chunk_id': 'chunk_000843',
              'chunk_char_len': 272,
              'embedding_model': 'solar-embedding-1-large',
              'tag_method': 'llm',
              'tag_confidence': 0.95}},
)
