from langchain_core.documents import Document

chunk = Document(
    page_content=('- 은 경우에는 제외합니다.)\n'
 '- 12. 목욕비용(약용 및 처방샴푸 값 포함) 및 귀세정제(이어클리너), 예방 가능한\n'
 '- 기생충(벼룩, 젝켄, 모공충 등)의 제거비용 및 기생충으로 발생한 질병의 치료\n'
 '- 비\n'
 '- 13. 펫호텔 비용 또는 위탁료, 산책료, 카운슬링 비용, 상담료, 지도료 및 이와 동\n'
 '- 종의 비용\n'
 '- 14. 왕진료, 가입동물의 이송비, 동물병원에 가지 않고 약제만 배달되는 배달료 및\n'
 '- 이와 동종의 비용\n'
 '- 15. 안락사 비용, 시체처치 및 해부검사, 장례비, 이장비 등 사후에 필요한 비용'),
    metadata={'source_doc': {'total_pages': 113},
 'doc': {'doc_type': 'terms',
         'file_name': 'kb_17390_gold_pet_dog_54_72_74_166.pdf',
         'insurer_code': 'kb',
         'product_code': '17390',
         'product_name': '[일반보험] KB반려행복펫보험',
         'total_pages': 1,
         'page': 1},
 'term_type': 'unknown',
 'clause': {'clause_type': 'exclusion', 'risk_domains': ['digestive']},
 'indexing': {'chunk_id': 'chunk_000607',
              'chunk_char_len': 284,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'normal',
              'tag_method': 'rule',
              'tag_confidence': 0.85}},
)
