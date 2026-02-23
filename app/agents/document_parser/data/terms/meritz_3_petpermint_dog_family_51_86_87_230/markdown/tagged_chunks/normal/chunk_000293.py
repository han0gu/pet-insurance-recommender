from langchain_core.documents import Document

chunk = Document(
    page_content=('- 법, 아로마테라피 등의 대체의료, 재활치료\n'
 '- ⑬ 목욕비용(약욕 및 처방샴푸 값 포함) 및 귀 세정제\n'
 '- (이어 클리너), 예방 가능한 기생충(벼룩, 진드기,\n'
 '- 모낭충 등)의 제거비용 및 기생충으로 발생한 질병의\n'
 '- 치료비\n'
 '- ⑭ 반려동물호텔 또는 보관 비용, 산책료, 카운슬링 비\n'
 '- 용, 상담 수수료, 지도 비용 및 이와 동종의 비용\n'
 '- ⑮ 왕진 비용, 가입동물의 이송비, 동물병원에 가지 않\n'
 '- 고 약제만 배달되는 배달료 및 이와 동종의 비용\n'
 '- ⑯ 안락사 비용, 시체처치 및 해부검사, 장례비, 이장비'),
    metadata={'source_doc': {'total_pages': 180},
 'doc': {'doc_type': 'terms',
         'file_name': 'meritz_3_petpermint_dog_family_51_86_87_230.pdf',
         'insurer_code': 'meritz',
         'product_code': '3',
         'product_name': '무배당 펫퍼민트 Puppy&Family보험 다이렉트2601',
         'total_pages': 1,
         'page': 1},
 'term_type': 'unknown',
 'clause': {'clause_type': 'other', 'risk_domains': ['digestive']},
 'indexing': {'chunk_id': 'chunk_000293',
              'chunk_char_len': 287,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'normal',
              'tag_method': 'rule',
              'tag_confidence': 0.25}},
)
