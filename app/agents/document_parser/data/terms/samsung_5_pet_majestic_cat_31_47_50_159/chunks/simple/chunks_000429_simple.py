from langchain_core.documents import Document

chunk = Document(
    page_content=('<용어풀이>\n'
 '[신의료기술평가위원회]\n'
 '의료법 제54조(신의료기술평가위원회의 설치 등)에 의거 설치된 위원회로서 신의료기술에 관한 최고의 심의기구를 말합니다.\n'
 '③ 제1항의 수술에서 아래에 정한 사항은 제외합니다.\n'
 '1. 흡인(吸引, 주사기 등으로 빨아들이는 것) 2. 천자(穿刺, 바늘 또는 관을 꽂아 체액 · 조직을 뽑아내거나 약물을 주입하는 것) '
 '등 의 조치 3. 신경(神經) BLOCK(신경의 차단) 4. 상해 원인 외 단순 미용성형 목적의 수술'),
    metadata={'source_doc': {'total_pages': 129},
 'doc': {'doc_type': 'terms',
         'file_name': 'samsung_5_pet_majestic_cat_31_47_50_159.pdf',
         'insurer_code': 'samsung',
         'product_code': '5',
         'product_name': '(장기)무배당 삼성화재 펫보험 의기냥냥',
         'total_pages': 107,
         'page': 79},
 'term_type': 'special',
 'clause': {'clause_type': 'exclusion', 'risk_domains': ['digestive', 'head']},
 'indexing': {'chunk_id': 'chunk_000429',
              'chunk_char_len': 249,
              'embedding_model': 'solar-embedding-1-large',
              'tag_method': 'rule',
              'tag_confidence': 0.85}},
)
