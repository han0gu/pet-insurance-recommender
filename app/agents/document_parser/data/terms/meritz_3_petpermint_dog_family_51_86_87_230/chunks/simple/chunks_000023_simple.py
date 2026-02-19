from langchain_core.documents import Document

chunk = Document(
    page_content=('① 전문등반(전문적인 등산용구를 사용하여 암벽 또는 빙 벽을 오르내리거나 특수한 기술, 경험, 사전훈련을 필 요로 하는 등반을 '
 '말합니다), 글라이더 조종, 스카이 다이빙, 스쿠버다이빙, 행글라이딩, 수상보트, 패러글 라이딩 ② 모터보트, 자동차 또는 오토바이에 '
 '의한 경기, 시범, 흥행(이를 위한 연습을 포함합니다) 또는 시운전(다 만, 공용도로상에서 시운전을 하는 동안 보험금 지급 사유가 발생한 '
 '경우에는 보장합니다) ③ 선박에 탑승하는 것을 직무로 하는 사람이 직무상 선 박에 탑승하고 있는 동안\n'
 '제6조(보험금 지급사유의 통지)'),
    metadata={'source_doc': {'total_pages': 180},
 'doc': {'doc_type': 'terms',
         'file_name': 'meritz_3_petpermint_dog_family_51_86_87_230.pdf',
         'insurer_code': 'meritz',
         'product_code': '3',
         'product_name': '무배당 펫퍼민트 Puppy&Family보험 다이렉트2601',
         'total_pages': 180,
         'page': 56},
 'term_type': 'basic',
 'clause': {'clause_type': 'coverage',
            'risk_domains': ['digestive', 'urinary']},
 'indexing': {'chunk_id': 'chunk_000023',
              'chunk_char_len': 297,
              'embedding_model': 'solar-embedding-1-large',
              'tag_method': 'rule',
              'tag_confidence': 0.8}},
)
