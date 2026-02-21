from langchain_core.documents import Document

chunk = Document(
    page_content=('. 다만, 뇌사판정을 받은 경우가 아닌 식<br>물인간상태(의식이 전혀 없고 사지의 자발적인 움직임<br>이 불가능하여 일상생활에서 항시 '
 '간호가 필요한 상<br>태)는 각 신체부위별 판정기준에 따라 평가한다.<br>5) 장해진단서에는 ① 장해진단명 및 발생시기 ② '
 '장해의<br>내용과 그 정도③ 사고와의 인과관계 및 사고의 관여<br>도 ④ 향후 치료의 문제 및 호전도를 필수적으로 기<br>재해야 '
 '한다'),
    metadata={'source_doc': {'total_pages': 160},
 'doc': {'doc_type': 'terms',
         'file_name': 'meritz_2_petpermint_cat_family_45_82_83_206.pdf',
         'insurer_code': 'meritz',
         'product_code': '2',
         'product_name': '무배당 펫퍼민트 Cat&Family보험 다이렉트2601',
         'total_pages': 1,
         'page': 1},
 'term_type': 'unknown',
 'clause': {'clause_type': 'other', 'risk_domains': ['digestive', 'head']},
 'indexing': {'chunk_id': 'chunk_000916',
              'chunk_char_len': 225,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'normal',
              'tag_method': 'rule',
              'tag_confidence': 0.25}},
)
