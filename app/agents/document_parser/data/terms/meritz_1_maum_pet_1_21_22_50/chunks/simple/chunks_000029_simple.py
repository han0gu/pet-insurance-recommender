from langchain_core.documents import Document

chunk = Document(
    page_content=('. 반려동물호텔 또는 보관 비용, 산책료, 카운슬링 비용, 상담 수수료, 지도 비용 및 이와 동종의 비용 17. 왕진 비용, 가입동물의 '
 '이송비, 동물병원에 가지 않고 약제만 배달되는 배달료 및 이와 동종의 비용 18. 안락사 비용, 시체처치 및 해부검사, 장례비, 이장비 '
 '등 사후에 필요한 비용 19. 마이크로 칩 이식 비용, 각종 증빙서류의 작성비용(우송비 포함) 20. 과잉진료행위로 인한 비용 21. '
 '아포퀠(Apoquel) 등의 JAK inhibitor(Janus kinase inhibitor) 약물 22'),
    metadata={'source_doc': {'total_pages': 50},
 'doc': {'doc_type': 'terms',
         'file_name': 'meritz_1_maum_pet_1_21_22_50.pdf',
         'insurer_code': 'meritz',
         'product_code': '1',
         'product_name': '메리츠 마음든든 반려동물보험',
         'total_pages': 50,
         'page': 5},
 'term_type': 'basic',
 'clause': {'clause_type': 'exclusion', 'risk_domains': ['other']},
 'indexing': {'chunk_id': 'chunk_000029',
              'chunk_char_len': 286,
              'embedding_model': 'solar-embedding-1-large',
              'tag_method': 'llm',
              'tag_confidence': 0.95}},
)
