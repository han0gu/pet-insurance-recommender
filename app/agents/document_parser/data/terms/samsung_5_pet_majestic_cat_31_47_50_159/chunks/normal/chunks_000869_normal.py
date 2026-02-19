from langchain_core.documents import Document

chunk = Document(
    page_content=('. 다만, 뇌사판정을 받은 경우가 아닌 식물인간상태(의식이 전혀 없고 사지의 자발적인 움직임이 불가능하여 일상생활에서 항시 간호가 필요한 '
 '상태)는 각 신체부위별 판정기준에 따라 평가한다. 마. 장해진단서에는 ① 장해진단명 및 발생시기 ② 장해의 내용과 그 정도 ③ 사고와 의 '
 '인과관계 및 사고의 관여도 ④ 향후 치료의 문제 및 호전도를 필수적으로 기 재해야 한다. 다만, 신경계 · 정신행동 장해의 경우 ① '
 '개호(장해로 혼자서 활동이 어려운 사람을 곁에서 돌보는 것) 여부 ② 객관적 이유 및 개호의 내용을 추가로 기재하여야 한다.'),
    metadata={'source_doc': {'total_pages': 129},
 'doc': {'doc_type': 'terms',
         'file_name': 'samsung_5_pet_majestic_cat_31_47_50_159.pdf',
         'insurer_code': 'samsung',
         'product_code': '5',
         'product_name': '(장기)무배당 삼성화재 펫보험 의기냥냥',
         'total_pages': 107,
         'page': 137},
 'term_type': 'special',
 'clause': {'clause_type': 'definition',
            'risk_domains': ['head', 'joint', 'other']},
 'indexing': {'chunk_id': 'chunk_000869',
              'chunk_char_len': 297,
              'embedding_model': 'solar-embedding-1-large',
              'tag_method': 'llm',
              'tag_confidence': 0.95}},
)
