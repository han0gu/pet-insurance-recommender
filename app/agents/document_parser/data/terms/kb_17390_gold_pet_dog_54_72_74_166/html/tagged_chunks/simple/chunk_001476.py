from langchain_core.documents import Document

chunk = Document(
    page_content=('. 다만, 뇌사판정을 받은 경우가 아닌 식물인간상태(의식<br>이 전혀 없고 사지의 자발적인 움직임이 불가능하여 일상생활에서 항시 '
 '간호<br>가 필요한 상태)는 각 신체부위별 판정기준에 따라 평가한다.<br>5) 장해진단서에는 ① 장해진단명 및 발생시기 ② 장해의 '
 '내용과 그 정도 ③ 사<br>고와의 인과관계 및 사고의 관여도 ④ 향후 치료의 문제 및 호전도를 필수적<br>으로 기재해야 한다'),
    metadata={'source_doc': {'total_pages': 113},
 'doc': {'doc_type': 'terms',
         'file_name': 'kb_17390_gold_pet_dog_54_72_74_166.pdf',
         'insurer_code': 'kb',
         'product_code': '17390',
         'product_name': '[일반보험] KB반려행복펫보험',
         'total_pages': 1,
         'page': 1},
 'term_type': 'unknown',
 'clause': {'clause_type': 'other', 'risk_domains': ['digestive', 'head']},
 'indexing': {'chunk_id': 'chunk_001476',
              'chunk_char_len': 219,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'simple',
              'tag_method': 'rule',
              'tag_confidence': 0.25}},
)
