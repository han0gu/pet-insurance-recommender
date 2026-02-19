from langchain_core.documents import Document

chunk = Document(
    page_content=('2. 임신, 출산(제왕절개를 포함합니다.), 인공유산과 관련된 비용 및 출산 후 증상 치 료 비용 3. 중성화, 불임 및 피임을 목적으로 '
 '한 수술 및 처치에 따른 비용 4. 산후 문제행동, 수유에 따르는 칼슘 부족에 의한 경련 및 기타 임신 · 출산과 관련 된 질병 치료에 '
 '대한 비용 5. 손톱의 절제(며느리발톱의 제거 포함), 잔존유치, 잠복고환, 배꼽허니아(배꼽부위탈장), 항문낭 제거 등 건강동물에 '
 '실시하는 외과수술 및 기타 검사 또는 손톱깎기 등의 처치비용 6. 미용으로 인한 비용 7'),
    metadata={'source_doc': {'total_pages': 107},
 'doc': {'doc_type': 'terms',
         'file_name': 'samsung_4_pet_majestic_dog_33_49_52_139.pdf',
         'insurer_code': 'samsung',
         'product_code': '4',
         'product_name': '(장기)무배당 삼성화재 펫보험 위풍댕댕',
         'total_pages': 91,
         'page': 101},
 'term_type': 'special',
 'clause': {'clause_type': 'exclusion', 'risk_domains': ['other']},
 'indexing': {'chunk_id': 'chunk_000554',
              'chunk_char_len': 275,
              'embedding_model': 'solar-embedding-1-large',
              'tag_method': 'llm',
              'tag_confidence': 0.95}},
)
