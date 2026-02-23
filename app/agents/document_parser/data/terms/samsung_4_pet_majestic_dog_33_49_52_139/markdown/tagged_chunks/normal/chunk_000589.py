from langchain_core.documents import Document

chunk = Document(
    page_content=('- 3. 미용성형 목적의 수술\n'
 '- 4. 검사 및 진단을 위한 수술 (생검, 복강경 검사)\n'
 '- ④ 제1항 내지 제3항에도 불구하고 이 특별약관의 보험계약일부터 그 날을 포함하여 1년\n'
 '- 이내에 발생한 슬관절탈구, 고관절탈구, 슬관절형성부전, 고관절형성부전 또는 기타\n'
 '- 이들과 유사한 사고에 대해서는 보험금을 지급하지 않습니다. 단, 이 계약이 제27조 (\n'
 '- 특별약관의 재가입에 관한 사항) 제1항 및 제2항에 따라 재가입하는 경우 또는 제27\n'
 '- 조 (특별약관의 재가입에 관한 사항) 제5항에 따라 보험계약이 연장된 경우에는 적용'),
    metadata={'source_doc': {'total_pages': 107},
 'doc': {'doc_type': 'terms',
         'file_name': 'samsung_4_pet_majestic_dog_33_49_52_139.pdf',
         'insurer_code': 'samsung',
         'product_code': '4',
         'product_name': '(장기)무배당 삼성화재 펫보험 위풍댕댕',
         'total_pages': 1,
         'page': 1},
 'term_type': 'unknown',
 'clause': {'clause_type': 'exclusion', 'risk_domains': ['digestive', 'joint']},
 'indexing': {'chunk_id': 'chunk_000589',
              'chunk_char_len': 297,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'normal',
              'tag_method': 'rule',
              'tag_confidence': 0.85}},
)
