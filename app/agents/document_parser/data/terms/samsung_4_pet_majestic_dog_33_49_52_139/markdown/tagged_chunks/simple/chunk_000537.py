from langchain_core.documents import Document

chunk = Document(
    page_content=('- 피보험자에게 발생된 손해에 대하여 관계 법령 등에 따라 손해배상의 책임을 집니다.\n'
 '- ② 회사는 보험금 지급 거절 및 지연지급의 사유가 없음을 알았거나 알 수 있었는데도 소\n'
 '- 를 제기하여 계약자 또는 피보험자에게 손해를 가한 경우에는 그에 따른 손해를 배상\n'
 '- 할 책임을 집니다.\n'
 '- ③ 회사가 보험금 지급여부 및 지급금액에 관하여 현저하게 공정을 잃은 합의로 계약자\n'
 '- 또는 피보험자에게 손해를 가한 경우에도 회사는 제2항에 따라 손해를 배상할 책임을\n'
 '- 집니다.'),
    metadata={'source_doc': {'total_pages': 107},
 'doc': {'doc_type': 'terms',
         'file_name': 'samsung_4_pet_majestic_dog_33_49_52_139.pdf',
         'insurer_code': 'samsung',
         'product_code': '4',
         'product_name': '(장기)무배당 삼성화재 펫보험 위풍댕댕',
         'total_pages': 1,
         'page': 1},
 'term_type': 'unknown',
 'clause': {'clause_type': 'coverage', 'risk_domains': ['other']},
 'indexing': {'chunk_id': 'chunk_000537',
              'chunk_char_len': 263,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'simple',
              'tag_method': 'rule',
              'tag_confidence': 0.8}},
)
