from langchain_core.documents import Document

chunk = Document(
    page_content=('- (Laser)를 이용하여 생체에 절단, 절제 등의 조작을 가하는 것도 포함됩니다.\n'
 '<용어풀이># [신의료기술평가위원회]의료법 제54조(신의료기술평가위원회의 설치 등)에 의거 설치된 위원회로서 신의료기술에\n'
 '관한 최고의 심의기구를 말합니다.# ③ 제1항의 수술에서 아래에 정한 사항은 제외합니다.- 1. 흡인(吸引, 주사기 등으로 빨아들이는 '
 '것)\n'
 '- 2. 천자(穿刺, 바늘 또는 관을 꽂아 체액 · 조직을 뽑아내거나 약물을 주입하는 것) 등\n'
 '- 의 조치\n'
 '- 3. 신경(神經) BLOCK(신경의 차단)'),
    metadata={'source_doc': {'total_pages': 129},
 'doc': {'doc_type': 'terms',
         'file_name': 'samsung_5_pet_majestic_cat_31_47_50_159.pdf',
         'insurer_code': 'samsung',
         'product_code': '5',
         'product_name': '(장기)무배당 삼성화재 펫보험 의기냥냥',
         'total_pages': 1,
         'page': 1},
 'term_type': 'unknown',
 'clause': {'clause_type': 'exclusion', 'risk_domains': ['digestive', 'head']},
 'indexing': {'chunk_id': 'chunk_000360',
              'chunk_char_len': 279,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'simple',
              'tag_method': 'rule',
              'tag_confidence': 0.85}},
)
