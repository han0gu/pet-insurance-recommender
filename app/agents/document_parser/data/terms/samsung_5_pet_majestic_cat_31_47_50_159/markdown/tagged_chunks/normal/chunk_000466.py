from langchain_core.documents import Document

chunk = Document(
    page_content=('- 진, 예방적 검사를 위한 비용\n'
 '- 3. 임신, 출산(제왕절개를 포함합니다), 인공유산과 관련된 비용 및 출산 후 증상 치료\n'
 '- 비용\n'
 '- 4. 중성화, 불임 및 피임을 목적으로 한 수술 및 처치에 따른 비용\n'
 '- 5. 손톱의 절제(며느리발톱의 제거 포함), 잔존유치, 잠복고환,\n'
 '- 배꼽허니아(배꼽부위탈장), 항문낭 제거 등 건강동물에 실시하는 외과수술 및 기타\n'
 '- 검사 또는 손톱깎기 등의 처치비용\n'
 '- 6. 미용으로 인한 비용\n'
 '- 7. 귀 성형, 꼬리 성형, 성대 제거 및 미용성형 등 질병치료가 아닌 수술에 소요되는\n'
 '- 비용'),
    metadata={'source_doc': {'total_pages': 129},
 'doc': {'doc_type': 'terms',
         'file_name': 'samsung_5_pet_majestic_cat_31_47_50_159.pdf',
         'insurer_code': 'samsung',
         'product_code': '5',
         'product_name': '(장기)무배당 삼성화재 펫보험 의기냥냥',
         'total_pages': 1,
         'page': 1},
 'term_type': 'unknown',
 'clause': {'clause_type': 'other', 'risk_domains': ['digestive']},
 'indexing': {'chunk_id': 'chunk_000466',
              'chunk_char_len': 295,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'normal',
              'tag_method': 'rule',
              'tag_confidence': 0.25}},
)
