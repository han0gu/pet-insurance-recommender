from langchain_core.documents import Document

chunk = Document(
    page_content=('5. 피임(避妊) 목적의 수술 6. 검사 및 진단을 위한 수술(생검(生検), 복강경검사(腹腔鏡検査) 등) 7. 기타 수술의 정의에 '
 '해당하지 않는 시술\n'
 '<예시안내>'),
    metadata={'source_doc': {'total_pages': 129},
 'doc': {'doc_type': 'terms',
         'file_name': 'samsung_5_pet_majestic_cat_31_47_50_159.pdf',
         'insurer_code': 'samsung',
         'product_code': '5',
         'product_name': '(장기)무배당 삼성화재 펫보험 의기냥냥',
         'total_pages': 107,
         'page': 80},
 'term_type': 'special',
 'clause': {'clause_type': 'definition', 'risk_domains': ['digestive']},
 'indexing': {'chunk_id': 'chunk_000430',
              'chunk_char_len': 90,
              'embedding_model': 'solar-embedding-1-large',
              'tag_method': 'rule',
              'tag_confidence': 0.55}},
)
