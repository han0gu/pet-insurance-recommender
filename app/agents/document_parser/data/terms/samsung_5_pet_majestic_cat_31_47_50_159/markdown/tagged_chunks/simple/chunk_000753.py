from langchain_core.documents import Document

chunk = Document(
    page_content=('| 7) 씹어먹는 기능 또는 말하는 기능에 약간의 장해를 남긴 때 | 5 |\n'
 '| 8) 치아에 14개 이상의 결손이 생긴 때 | 20 |\n'
 '| 9) 치아에 7개 이상의 결손이 생긴 때 | 10 |\n'
 '| 10) 치아에 5개 이상의 결손이 생긴 때 | 5 |\n'
 '# 나. 장해의 평가기준- 1) 씹어먹는 기능의 장해는 윗니(상악치아)와 아랫니(하악치아)의 맞물림(교합), 배\n'
 '- 열상태 및 아래턱의 개구운동, 삼킴(연하)운동 등에 따라 종합적으로 판단하여\n'
 '- 결정한다.'),
    metadata={'source_doc': {'total_pages': 129},
 'doc': {'doc_type': 'terms',
         'file_name': 'samsung_5_pet_majestic_cat_31_47_50_159.pdf',
         'insurer_code': 'samsung',
         'product_code': '5',
         'product_name': '(장기)무배당 삼성화재 펫보험 의기냥냥',
         'total_pages': 1,
         'page': 1},
 'term_type': 'unknown',
 'clause': {'clause_type': 'other', 'risk_domains': ['dental', 'digestive']},
 'indexing': {'chunk_id': 'chunk_000753',
              'chunk_char_len': 255,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'simple',
              'tag_method': 'rule',
              'tag_confidence': 0.25}},
)
