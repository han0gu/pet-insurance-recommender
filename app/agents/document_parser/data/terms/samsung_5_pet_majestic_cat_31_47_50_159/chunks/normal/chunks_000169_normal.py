from langchain_core.documents import Document

chunk = Document(
    page_content=('임을 집니다.\n'
 '② 회사는 보험금 지급 거절 및 지연지급의 사유가 없음을 알았거나 알 수 있었는데도 소 를 제기하여 계약자, 피보험자 또는 보험수익자에게 '
 '손해를 가한 경우에는 그에 따른 손해를 배상할 책임을 집니다. ③ 회사가 보험금 지급여부 및 지급금액에 관하여 현저하게 공정을 잃은 '
 '합의로 보험수 익자에게 손해를 가한 경우에도 회사는 제2항에 따라 손해를 배상할 책임을 집니다.\n'
 '<용어풀이>\n'
 '[현저하게 공정을 잃은 합의]'),
    metadata={'source_doc': {'total_pages': 129},
 'doc': {'doc_type': 'terms',
         'file_name': 'samsung_5_pet_majestic_cat_31_47_50_159.pdf',
         'insurer_code': 'samsung',
         'product_code': '5',
         'product_name': '(장기)무배당 삼성화재 펫보험 의기냥냥',
         'total_pages': 107,
         'page': 46},
 'term_type': 'basic',
 'clause': {'clause_type': 'coverage', 'risk_domains': ['other']},
 'indexing': {'chunk_id': 'chunk_000169',
              'chunk_char_len': 236,
              'embedding_model': 'solar-embedding-1-large',
              'tag_method': 'rule',
              'tag_confidence': 0.55}},
)
