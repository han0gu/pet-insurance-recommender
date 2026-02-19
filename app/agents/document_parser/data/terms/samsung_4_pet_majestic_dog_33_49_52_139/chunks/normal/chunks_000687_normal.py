from langchain_core.documents import Document

chunk = Document(
    page_content=('관리 하에 직접적인 치료를 목적으로 기구를 사용하여 생체에 절개, 절단, 절제 등의 조작을 가하 는 것을 말합니다. 단 수술에서 아래에 '
 '정한 사항은 제외합니다\n'
 '1. 흡인 (주사기 등으로 빨아 들이는 것) 2. 천자 (바늘 또는 관을 꽂아 체액, 조직을 뽑아내거나 약물을 주입하는 것) 등의 조치 '
 '3. 미용성형 목적의 수술 4. 검사 및 진단을 위한 수술 (생검, 복강경 검사)'),
    metadata={'source_doc': {'total_pages': 107},
 'doc': {'doc_type': 'terms',
         'file_name': 'samsung_4_pet_majestic_dog_33_49_52_139.pdf',
         'insurer_code': 'samsung',
         'product_code': '4',
         'product_name': '(장기)무배당 삼성화재 펫보험 위풍댕댕',
         'total_pages': 91,
         'page': 114},
 'term_type': 'special',
 'clause': {'clause_type': 'exclusion', 'risk_domains': ['digestive']},
 'indexing': {'chunk_id': 'chunk_000687',
              'chunk_char_len': 210,
              'embedding_model': 'solar-embedding-1-large',
              'tag_method': 'rule',
              'tag_confidence': 0.85}},
)
