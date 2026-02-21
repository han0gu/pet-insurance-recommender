from langchain_core.documents import Document

chunk = Document(
    page_content=('품 이외의 것(건강보조식품, 의약품지정이 되어 있지 않은 한방약, 의약부외품 등)\n'
 '카. 한의학(단, 침구는 제외합니다.), 인도의학, 허브요법, 아로마테라피 등의 대체의료\n'
 '타. 목욕비용(약욕 및 처방샹품 값 포함) 및 이어클리너, 벼룩, 젝켄, 모공충의 제거비용\n'
 '파. 펫호텔 비용 또는 위탁료, 산책료, 카운슬링 비용, 상담료, 지도료 및 이와 동종의 비용\n'
 '하. 왕진료, 가입동물의 이송비, 동물병원에 가지 않고 약제만 배달되는 배달료 및 이와 동종의'),
    metadata={'source_doc': {'total_pages': 35},
 'doc': {'doc_type': 'terms',
         'file_name': 'samsung_2_cat_anypet_3_20_21_37.pdf',
         'insurer_code': 'samsung',
         'product_code': '2',
         'product_name': '(일반)반려묘보험 애니펫',
         'total_pages': 1,
         'page': 1},
 'term_type': 'unknown',
 'clause': {'clause_type': 'exclusion', 'risk_domains': ['digestive']},
 'indexing': {'chunk_id': 'chunk_000017',
              'chunk_char_len': 252,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'normal',
              'tag_method': 'rule',
              'tag_confidence': 0.85}},
)
