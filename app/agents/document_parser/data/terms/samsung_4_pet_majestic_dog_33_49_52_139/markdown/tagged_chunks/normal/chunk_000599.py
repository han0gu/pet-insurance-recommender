from langchain_core.documents import Document

chunk = Document(
    page_content=('- 료 비용\n'
 '- 3. 중성화, 불임 및 피임을 목적으로 한 수술 및 처치에 따른 비용\n'
 '- 4. 산후 문제행동, 수유에 따르는 칼슘 부족에 의한 경련 및 기타 임신ㆍ출산과 관련\n'
 '- 된 질병 치료에 대한 비용\n'
 '- 5. 손톱의 절제(며느리발톱의 제거 포함), 잔존유치, 잠복고환,\n'
 '- 배꼽허니아(배꼽부위탈장), 항문낭 제거 등 건강동물에 실시하는 외과수술 및 기타\n'
 '- 검사 또는 손톱깎기 등의 처치비용\n'
 '- 6. 미용으로 인한 비용\n'
 '- 7. 귀 성형, 꼬리 성형, 성대 제거 및 미용성형 등 질병치료가 아닌 수술에 소요되는\n'
 '- 비용'),
    metadata={'source_doc': {'total_pages': 107},
 'doc': {'doc_type': 'terms',
         'file_name': 'samsung_4_pet_majestic_dog_33_49_52_139.pdf',
         'insurer_code': 'samsung',
         'product_code': '4',
         'product_name': '(장기)무배당 삼성화재 펫보험 위풍댕댕',
         'total_pages': 1,
         'page': 1},
 'term_type': 'unknown',
 'clause': {'clause_type': 'other', 'risk_domains': ['digestive', 'head']},
 'indexing': {'chunk_id': 'chunk_000599',
              'chunk_char_len': 293,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'normal',
              'tag_method': 'rule',
              'tag_confidence': 0.25}},
)
