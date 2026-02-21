from langchain_core.documents import Document

chunk = Document(
    page_content=('외의 것(건강보조 식품, 의약품지정이 되어 있지 않은 한방약, 의약부외품 등)\n'
 '14. 한의학(단, 침술을 제외합니다.), 인도 의학, 허브요법, 아로마테라피 등의 대체의료,\n'
 '재활치료\n'
 '15. 목욕비용(약욕 및 처방샴푸 값 포함) 및 귀 세정제(이어 클리너), 예방 가능한 기생충(벼\n'
 '룩, 진드기, 모낭충 등)의 제거비용 및 기생충으로 발생한 질병의 치료비\n'
 '16. 반려동물호텔 또는 보관 비용, 산책료, 카운슬링 비용, 상담 수수료, 지도 비용 및\n'
 '이와 동종의 비용'),
    metadata={'source_doc': {'total_pages': 50},
 'doc': {'doc_type': 'terms',
         'file_name': 'meritz_1_maum_pet_1_21_22_50.pdf',
         'insurer_code': 'meritz',
         'product_code': '1',
         'product_name': '메리츠 마음든든 반려동물보험',
         'total_pages': 1,
         'page': 1},
 'term_type': 'unknown',
 'clause': {'clause_type': 'exclusion', 'risk_domains': ['other']},
 'indexing': {'chunk_id': 'chunk_000025',
              'chunk_char_len': 258,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'simple',
              'tag_method': 'rule',
              'tag_confidence': 0.85}},
)
