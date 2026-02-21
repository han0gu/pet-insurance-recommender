from langchain_core.documents import Document

chunk = Document(
    page_content=('관한 최고의 심의기구를 말합니다.# ③ 제1항의 수술에서 아래에 정한 사항은 제외합니다.- 1. 흡인(吸引, 주사기 등으로 빨아들이는 '
 '것)\n'
 '- 2. 천자(穿刺, 바늘 또는 관을 꽂아 체액 · 조직을 뽑아내거나 약물을 주입하는 것) 등\n'
 '- 의 조치\n'
 '- 3. 신경(神經) BLOCK(신경의 차단)\n'
 '- 4. 상해 원인 외 단순 미용성형 목적의 수술\n'
 '- 81 -5. 피임(避妊) 목적의 수술\n'
 '6. 검사 및 진단을 위한 수술(생검(生検), 복강경검사(腹腔鏡検査) 등)\n'
 '7. 기타 수술의 정의에 해당하지 않는 시술- \n'
 '| <예시안내> |'),
    metadata={'source_doc': {'total_pages': 107},
 'doc': {'doc_type': 'terms',
         'file_name': 'samsung_4_pet_majestic_dog_33_49_52_139.pdf',
         'insurer_code': 'samsung',
         'product_code': '4',
         'product_name': '(장기)무배당 삼성화재 펫보험 위풍댕댕',
         'total_pages': 1,
         'page': 1},
 'term_type': 'unknown',
 'clause': {'clause_type': 'exclusion', 'risk_domains': ['digestive', 'head']},
 'indexing': {'chunk_id': 'chunk_000362',
              'chunk_char_len': 294,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'normal',
              'tag_method': 'rule',
              'tag_confidence': 0.85}},
)
