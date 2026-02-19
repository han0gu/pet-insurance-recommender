from langchain_core.documents import Document

chunk = Document(
    page_content=('1. 전문등반(전문적인 등산용구를 사용하여 암벽 또는 빙벽을 오르내리거나 특수한 기 술, 경험, 사전훈련을 필요로 하는 등반을 '
 '말합니다), 글라이더 조종, 스카이다이 빙, 스쿠버다이빙, 행글라이딩, 수상보트, 패러글라이딩 2. 모터보트, 자동차 또는 오토바이에 '
 '의한 경기, 시범, 흥행(이를 위한 연습을 포함합 니다) 또는 시운전(다만, 공용도로상에서 시운전을 하는 동안 보험금 지급사유가 발생한 '
 '경우에는 보장합니다) 3. 선박에 탑승하는 것을 직무로 하는 사람이 직무상 선박에 탑승하고 있는 동안\n'
 '<용어풀이>\n'
 '[흥행]'),
    metadata={'source_doc': {'total_pages': 107},
 'doc': {'doc_type': 'terms',
         'file_name': 'samsung_4_pet_majestic_dog_33_49_52_139.pdf',
         'insurer_code': 'samsung',
         'product_code': '4',
         'product_name': '(장기)무배당 삼성화재 펫보험 위풍댕댕',
         'total_pages': 91,
         'page': 53},
 'term_type': 'special',
 'clause': {'clause_type': 'coverage',
            'risk_domains': ['digestive', 'urinary']},
 'indexing': {'chunk_id': 'chunk_000192',
              'chunk_char_len': 290,
              'embedding_model': 'solar-embedding-1-large',
              'tag_method': 'rule',
              'tag_confidence': 0.8}},
)
