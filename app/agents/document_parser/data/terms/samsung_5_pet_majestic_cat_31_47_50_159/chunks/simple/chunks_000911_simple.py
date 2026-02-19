from langchain_core.documents import Document

chunk = Document(
    page_content=('. 척추체(척추뼈 몸통)의 압박률은 인접 상 · 하부[인접 상 + 하부 척추체(척추뼈 몸통)에 진구성 골절이 있거나, 다발성 척추골절이 '
 '있는 경우에는 골절된 척추와 가장 인접한 상 + 하부] 정상 척추체(척추뼈 몸통)의 전방 높이의 평균에 대한 골절된 척추체(척추뼈 몸통) '
 '전방 높이의 감소비를 압박률로 정한다. 다) 척추(등뼈)의 기형장해는 「산업재해보상보험법 시행규칙」 상 경추부, 흉추 부, 요추부로 '
 '구분하여 각각을 하나의 운동단위로 보며, 하나의 운동단위'),
    metadata={'source_doc': {'total_pages': 129},
 'doc': {'doc_type': 'terms',
         'file_name': 'samsung_5_pet_majestic_cat_31_47_50_159.pdf',
         'insurer_code': 'samsung',
         'product_code': '5',
         'product_name': '(장기)무배당 삼성화재 펫보험 의기냥냥',
         'total_pages': 107,
         'page': 140},
 'term_type': 'special',
 'clause': {'clause_type': 'coverage', 'risk_domains': ['digestive']},
 'indexing': {'chunk_id': 'chunk_000911',
              'chunk_char_len': 258,
              'embedding_model': 'solar-embedding-1-large',
              'tag_method': 'rule',
              'tag_confidence': 0.8}},
)
