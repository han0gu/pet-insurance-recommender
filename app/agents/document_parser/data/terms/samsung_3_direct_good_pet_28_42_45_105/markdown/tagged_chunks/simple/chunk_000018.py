from langchain_core.documents import Document

chunk = Document(
    page_content=('- 술, 경험, 사전훈련을 필요로 하는 등반을 말합니다), 글라이더 조종, 스카이다이\n'
 '- 빙, 스쿠버다이빙, 행글라이딩, 수상보트, 패러글라이딩\n'
 '- 2. 모터보트, 자동차 또는 오토바이에 의한 경기, 시범, 흥행(이를 위한 연습을 포함합\n'
 '- 니다) 또는 시운전(다만, 공용도로상에서 시운전을 하는 동안 보험금 지급사유가\n'
 '- 발생한 경우에는 보장합니다)\n'
 '- 3. 선박에 탑승하는 것을 직무로 하는 사람이 직무상 선박에 탑승하고 있는 동안'),
    metadata={'source_doc': {'total_pages': 78},
 'doc': {'doc_type': 'terms',
         'file_name': 'samsung_3_direct_good_pet_28_42_45_105.pdf',
         'insurer_code': 'samsung',
         'product_code': '3',
         'product_name': '(장기)무배당 삼성화재 다이렉트 '
                         '착한펫보험(강아지)',
         'total_pages': 1,
         'page': 1},
 'term_type': 'unknown',
 'clause': {'clause_type': 'coverage',
            'risk_domains': ['digestive', 'urinary']},
 'indexing': {'chunk_id': 'chunk_000018',
              'chunk_char_len': 243,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'simple',
              'tag_method': 'rule',
              'tag_confidence': 0.8}},
)
