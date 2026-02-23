from langchain_core.documents import Document

chunk = Document(
    page_content=('- 2. 천자(穿刺, 바늘 또는 관을 꽂아 체액 · 조직을 뽑아내거나 약물을 주입하는 것) 등\n'
 '- 의 조치\n'
 '- 3. 신경(神經) BLOCK(신경의 차단)\n'
 '- 4. 미용성형 목적의 수술\n'
 '- 5. 피임(避妊) 목적의 수술\n'
 '- 6. 검사 및 진단을 위한 수술(생검(生機), 복강경검사(腹腔鏡検査) 등)\n'
 '- 7. 기타 수술의 정의에 해당하지 않는 시술\n'
 '<예시안내>[기타 수술의 정의에 해당하지 않는 시술]- - 체외 충격파 쇄석술\n'
 '- - 변연절제를 동반하지 않은 단순 창상봉합술\n'
 '- - 절개, 배농 또는 도관삽입술'),
    metadata={'source_doc': {'total_pages': 129},
 'doc': {'doc_type': 'terms',
         'file_name': 'samsung_5_pet_majestic_cat_31_47_50_159.pdf',
         'insurer_code': 'samsung',
         'product_code': '5',
         'product_name': '(장기)무배당 삼성화재 펫보험 의기냥냥',
         'total_pages': 1,
         'page': 1},
 'term_type': 'unknown',
 'clause': {'clause_type': 'other', 'risk_domains': ['digestive', 'head']},
 'indexing': {'chunk_id': 'chunk_000329',
              'chunk_char_len': 283,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'simple',
              'tag_method': 'rule',
              'tag_confidence': 0.25}},
)
