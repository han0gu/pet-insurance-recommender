from langchain_core.documents import Document

chunk = Document(
    page_content=('- 태)는 각 신체부위별 판정기준에 따라 평가한다.\n'
 '- 5) 장해진단서에는 ① 장해진단명 및 발생시기 ② 장해의\n'
 '- 내용과 그 정도③ 사고와의 인과관계 및 사고의 관여\n'
 '- 도 ④ 향후 치료의 문제 및 호전도를 필수적으로 기\n'
 '- 재해야 한다. 다만, 신경계․정신행동 장해의 경우 ①\n'
 '- 개호(장해로 혼자서 활동이 어려운 사람을 곁에서 돌\n'
 '- 보는 것) 여부 ② 객관적 이유 및 개호의 내용을 추가\n'
 '- 로 기재하여야 한다.\n'
 '# \uf000 장해분류별 판정기준# 1. 눈의 장해# 가. 장해의 분류| 장해의 분류 | 지급률 |'),
    metadata={'source_doc': {'total_pages': 180},
 'doc': {'doc_type': 'terms',
         'file_name': 'meritz_3_petpermint_dog_family_51_86_87_230.pdf',
         'insurer_code': 'meritz',
         'product_code': '3',
         'product_name': '무배당 펫퍼민트 Puppy&Family보험 다이렉트2601',
         'total_pages': 1,
         'page': 1},
 'term_type': 'unknown',
 'clause': {'clause_type': 'other',
            'risk_domains': ['digestive', 'eye', 'head']},
 'indexing': {'chunk_id': 'chunk_000590',
              'chunk_char_len': 286,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'simple',
              'tag_method': 'rule',
              'tag_confidence': 0.25}},
)
