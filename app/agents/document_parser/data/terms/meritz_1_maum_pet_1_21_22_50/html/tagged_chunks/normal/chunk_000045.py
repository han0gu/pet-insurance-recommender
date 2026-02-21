from langchain_core.documents import Document

chunk = Document(
    page_content=('. 목욕비용(약욕 및 처방샴푸 값 포함) 및 귀 세정제(이어 클리너), 예방 가능한 기생충(벼<br>룩, 진드기, 모낭충 등)의 제거비용 '
 '및 기생충으로 발생한 질병의 치료비<br>16. 반려동물호텔 또는 보관 비용, 산책료, 카운슬링 비용, 상담 수수료, 지도 비용 '
 '및<br>이와 동종의 비용<br>17. 왕진 비용, 가입동물의 이송비, 동물병원에 가지 않고 약제만 배달되는 배달료 및<br>이와 동종의 '
 '비용<br>18. 안락사 비용, 시체처치 및 해부검사, 장례비, 이장비 등 사후에 필요한 비용<br>19'),
    metadata={'source_doc': {'total_pages': 50},
 'doc': {'doc_type': 'terms',
         'file_name': 'meritz_1_maum_pet_1_21_22_50.pdf',
         'insurer_code': 'meritz',
         'product_code': '1',
         'product_name': '메리츠 마음든든 반려동물보험',
         'total_pages': 1,
         'page': 1},
 'term_type': 'unknown',
 'clause': {'clause_type': 'other', 'risk_domains': ['digestive']},
 'indexing': {'chunk_id': 'chunk_000045',
              'chunk_char_len': 284,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'normal',
              'tag_method': 'rule',
              'tag_confidence': 0.25}},
)
