from langchain_core.documents import Document

chunk = Document(
    page_content=('이외의 것(건강보조 식<br>품, 의약품지정이 되어 있지 않은 한방약, 의약부외<br>품 등)<br>⑫ 한의학(단, 침술을 '
 '제외합니다.), 인도 의학, 허브요<br>법, 아로마테라피 등의 대체의료, 재활치료<br>⑬ 목욕비용(약욕 및 처방샴푸 값 포함) 및 귀 '
 '세정제<br>(이어 클리너), 예방 가능한 기생충(벼룩, 진드기,<br>모낭충 등)의 제거비용 및 기생충으로 발생한 '
 '질병의<br>치료비<br>⑭ 반려동물호텔 또는 보관 비용, 산책료, 카운슬링 비<br>용, 상담 수수료, 지도 비용 및 이와 동종의 '
 '비용<br>⑮ 왕진 비용,'),
    metadata={'source_doc': {'total_pages': 160},
 'doc': {'doc_type': 'terms',
         'file_name': 'meritz_2_petpermint_cat_family_45_82_83_206.pdf',
         'insurer_code': 'meritz',
         'product_code': '2',
         'product_name': '무배당 펫퍼민트 Cat&Family보험 다이렉트2601',
         'total_pages': 1,
         'page': 1},
 'term_type': 'unknown',
 'clause': {'clause_type': 'exclusion', 'risk_domains': ['other']},
 'indexing': {'chunk_id': 'chunk_000589',
              'chunk_char_len': 299,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'simple',
              'tag_method': 'rule',
              'tag_confidence': 0.85}},
)
