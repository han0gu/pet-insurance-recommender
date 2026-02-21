from langchain_core.documents import Document

chunk = Document(
    page_content=('- 5) 장해진단서에는 ① 장해진단명 및 발생시기 ② 장해의 내용과 그 정도 ③ 사\n'
 '- 고와의 인과관계 및 사고의 관여도 ④ 향후 치료의 문제 및 호전도를 필수적\n'
 '- 으로 기재해야 한다. 다만, 신경계․정신행동 장해의 경우 ① 개호(장해로 혼\n'
 '- 자서 활동이 어려운 사람을 곁에서 돌보는 것) 여부 ② 객관적 이유 및 개호\n'
 '- 의 내용을 추가로 기재하여야 한다.\n'
 '# \uf000 장해분류별 판정기준# 1. 눈의 장해# 가. 장해의 분류| 장해의 분류 | 지급률 |\n'
 '| --- | --- |\n'
 '| 1) 두 눈이 멀었을 때 | 100 |'),
    metadata={'source_doc': {'total_pages': 113},
 'doc': {'doc_type': 'terms',
         'file_name': 'kb_17390_gold_pet_dog_54_72_74_166.pdf',
         'insurer_code': 'kb',
         'product_code': '17390',
         'product_name': '[일반보험] KB반려행복펫보험',
         'total_pages': 1,
         'page': 1},
 'term_type': 'unknown',
 'clause': {'clause_type': 'other',
            'risk_domains': ['digestive', 'eye', 'head']},
 'indexing': {'chunk_id': 'chunk_000837',
              'chunk_char_len': 292,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'normal',
              'tag_method': 'rule',
              'tag_confidence': 0.25}},
)
