from langchain_core.documents import Document

chunk = Document(
    page_content=('이와 동종의 비용\n'
 '17. 왕진 비용, 가입동물의 이송비, 동물병원에 가지 않고 약제만 배달되는 배달료 및\n'
 '이와 동종의 비용\n'
 '18. 안락사 비용, 시체처치 및 해부검사, 장례비, 이장비 등 사후에 필요한 비용\n'
 '19. 마이크로 칩 이식 비용, 각종 증빙서류의 작성비용(우송비 포함)\n'
 '20. 과잉진료행위로 인한 비용\n'
 '21. 아포퀠(Apoquel) 등의 JAK inhibitor(Janus kinase inhibitor) 약물\n'
 '22. 스케일링, 발치 등을 포함한 치아의 치과치료비용(단, 치아를 제외한 구강질환만 보'),
    metadata={'source_doc': {'total_pages': 50},
 'doc': {'doc_type': 'terms',
         'file_name': 'meritz_1_maum_pet_1_21_22_50.pdf',
         'insurer_code': 'meritz',
         'product_code': '1',
         'product_name': '메리츠 마음든든 반려동물보험',
         'total_pages': 1,
         'page': 1},
 'term_type': 'unknown',
 'clause': {'clause_type': 'exclusion',
            'risk_domains': ['dental', 'digestive']},
 'indexing': {'chunk_id': 'chunk_000026',
              'chunk_char_len': 284,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'normal',
              'tag_method': 'rule',
              'tag_confidence': 0.85}},
)
