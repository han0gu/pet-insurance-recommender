from langchain_core.documents import Document

chunk = Document(
    page_content=('② 회사는 아래의 사유로 생긴 손해는 보상하지 않습니다.\n'
 '1. 질병을 원인으로 하지 않는 신체검사, 예방접종, 인공유산, 불임시술, 제왕절개수술 2. 피로, 권태, 심신허약 등을 치료하기 위한 '
 '안정치료 3. 위생관리, 미모를 위한 성형수술 4. 정상분만, 치과질환\n'
 '제7조 (특별약관의 소멸)'),
    metadata={'source_doc': {'total_pages': 107},
 'doc': {'doc_type': 'terms',
         'file_name': 'samsung_4_pet_majestic_dog_33_49_52_139.pdf',
         'insurer_code': 'samsung',
         'product_code': '4',
         'product_name': '(장기)무배당 삼성화재 펫보험 위풍댕댕',
         'total_pages': 91,
         'page': 127},
 'term_type': 'special',
 'clause': {'clause_type': 'exclusion', 'risk_domains': ['digestive']},
 'indexing': {'chunk_id': 'chunk_000803',
              'chunk_char_len': 162,
              'embedding_model': 'solar-embedding-1-large',
              'tag_method': 'rule',
              'tag_confidence': 0.85}},
)
