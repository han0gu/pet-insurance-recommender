from langchain_core.documents import Document

chunk = Document(
    page_content=('- 마. 장해진단서에는 ① 장해진단명 및 발생시기 ② 장해의 내용과 그 정도 ③ 사고와\n'
 '- 의 인과관계 및 사고의 관여도 ④ 향후 치료의 문제 및 호전도를 필수적으로 기\n'
 '- 재해야 한다. 다만, 신경계 · 정신행동 장해의 경우 ① 개호(장해로 혼자서 활동이\n'
 '- 어려운 사람을 곁에서 돌보는 것) 여부 ② 객관적 이유 및 개호의 내용을 추가로\n'
 '- 기재하여야 한다.\n'
 '<장해분류별 판정기준>- \n'
 '# 1. 눈의 장해# 가. 장해의 분류| 장 해 의 분 류 | 지급률(%) |\n'
 '| --- | --- |\n'
 '| 1) 두 눈이 멀었을 때 | 100 |'),
    metadata={'source_doc': {'total_pages': 129},
 'doc': {'doc_type': 'terms',
         'file_name': 'samsung_5_pet_majestic_cat_31_47_50_159.pdf',
         'insurer_code': 'samsung',
         'product_code': '5',
         'product_name': '(장기)무배당 삼성화재 펫보험 의기냥냥',
         'total_pages': 1,
         'page': 1},
 'term_type': 'unknown',
 'clause': {'clause_type': 'other',
            'risk_domains': ['digestive', 'eye', 'head']},
 'indexing': {'chunk_id': 'chunk_000734',
              'chunk_char_len': 299,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'normal',
              'tag_method': 'rule',
              'tag_confidence': 0.25}},
)
