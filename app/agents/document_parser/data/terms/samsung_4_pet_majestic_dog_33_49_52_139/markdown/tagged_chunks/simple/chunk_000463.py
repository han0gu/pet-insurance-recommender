from langchain_core.documents import Document

chunk = Document(
    page_content=('- 금을 지급하지 않는 사유) 제2항의 의료비 및 비용을 위한 검사는 제외합니다.\n'
 '- 1. 신체검사, 정형검사, 신경계검사, 안검사, 피부과검사 등 기본검사\n'
 '- 2. X-ray, 초음파검사, CT, MRI, 내시경검사 등 영상검사\n'
 '- 3. 혈액검사, 임상병리검사, 조직병리검사, 배양검사 등 실험실검사\n'
 '- ⑧ 제2항에도 불구하고 제27조 (특별약관의 재가입에 관한 사항) 제1항 및 제2항에 따라\n'
 '- 재가입하는 경우 또는 제27조 (특별약관의 재가입에 관한 사항) 제5항에 따라 보험계'),
    metadata={'source_doc': {'total_pages': 107},
 'doc': {'doc_type': 'terms',
         'file_name': 'samsung_4_pet_majestic_dog_33_49_52_139.pdf',
         'insurer_code': 'samsung',
         'product_code': '4',
         'product_name': '(장기)무배당 삼성화재 펫보험 위풍댕댕',
         'total_pages': 1,
         'page': 1},
 'term_type': 'unknown',
 'clause': {'clause_type': 'exclusion',
            'risk_domains': ['digestive', 'head', 'skin']},
 'indexing': {'chunk_id': 'chunk_000463',
              'chunk_char_len': 271,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'simple',
              'tag_method': 'rule',
              'tag_confidence': 0.85}},
)
