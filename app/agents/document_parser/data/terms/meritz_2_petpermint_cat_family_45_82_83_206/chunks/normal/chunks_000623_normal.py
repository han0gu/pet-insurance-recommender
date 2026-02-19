from langchain_core.documents import Document

chunk = Document(
    page_content=('. 4) 의학적으로 뇌사판정을 받고 호흡기능과 심장박동기 능을 상실하여 인공심박동기 등 장치에 의존하여 생명 을 연장하고 있는 뇌사상태는 '
 '장해의 판정대상에 포함 되지 않는다. 다만, 뇌사판정을 받은 경우가 아닌 식 물인간상태(의식이 전혀 없고 사지의 자발적인 움직임 이 '
 '불가능하여 일상생활에서 항시 간호가 필요한 상 태)는 각 신체부위별 판정기준에 따라 평가한다. 5) 장해진단서에는 ① 장해진단명 및 '
 '발생시기 ② 장해의 내용과 그 정도③ 사고와의 인과관계 및 사고의 관여 도 ④ 향후 치료의 문제 및 호전도를 필수적으로 기 재해야 한다'),
    metadata={'source_doc': {'total_pages': 160},
 'doc': {'doc_type': 'terms',
         'file_name': 'meritz_2_petpermint_cat_family_45_82_83_206.pdf',
         'insurer_code': 'meritz',
         'product_code': '2',
         'product_name': '무배당 펫퍼민트 Puppy&Family보험 다이렉트2601',
         'total_pages': 160,
         'page': 177},
 'term_type': 'special',
 'clause': {'clause_type': 'exclusion', 'risk_domains': ['head', 'other']},
 'indexing': {'chunk_id': 'chunk_000623',
              'chunk_char_len': 300,
              'embedding_model': 'solar-embedding-1-large',
              'tag_method': 'llm',
              'tag_confidence': 0.95}},
)
