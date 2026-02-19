from langchain_core.documents import Document

chunk = Document(
    page_content=('③ 제1항의 수술에서 아래에 정한 사항은 제외합니다.\n'
 '1. 흡인(吸引, 주사기 등으로 빨아들이는 것) 2. 천자(穿刺, 바늘 또는 관을 꽂아 체액·조직을 뽑아내거나 약물을 주입하는 것) 등 '
 '의 조치 3. 신경(神經) BLOCK(신경의 차단) 4. 상해 원인 외 단순 미용성형 목적의 수술 5. 피임(避妊) 목적의 수술 6. '
 '검사 및 진단을 위한 수술(생검(生檢), 복강경검사(腹腔鏡檢査) 등)'),
    metadata={'source_doc': {'total_pages': 129},
 'doc': {'doc_type': 'terms',
         'file_name': 'samsung_5_pet_majestic_cat_31_47_50_159.pdf',
         'insurer_code': 'samsung',
         'product_code': '5',
         'product_name': '(장기)무배당 삼성화재 펫보험 의기냥냥',
         'total_pages': 107,
         'page': 81},
 'term_type': 'special',
 'clause': {'clause_type': 'exclusion', 'risk_domains': ['digestive', 'head']},
 'indexing': {'chunk_id': 'chunk_000441',
              'chunk_char_len': 219,
              'embedding_model': 'solar-embedding-1-large',
              'tag_method': 'rule',
              'tag_confidence': 0.85}},
)
