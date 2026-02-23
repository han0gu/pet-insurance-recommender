from langchain_core.documents import Document

chunk = Document(
    page_content=('| 8) 한 눈에 뚜렷한 시야장해를 남긴 때 | 5 |\n'
 '| 9) 한눈의 눈꺼풀에 뚜렷한 결손을 남긴 때 | 10 |\n'
 '| 10) 한눈의 눈꺼풀에 뚜렷한 운동장해를 남긴 때 | 5 |\n'
 '177나. 장해판정기준- 1) 시력장해의 경우 공인된 시력검사표에 따라 최소 3회\n'
 '- 이상 측정한다.\n'
 '- 2) “교정시력”이라 함은 안경(콘택트렌즈를 포함한 모\n'
 '- 든 종류의 시력 교정수단)으로 교정한 원거리 최대교\n'
 '- 정시력을 말한다. 다만, 각막이식술을 받은 환자인 경\n'
 '- 우 각막이식술 이전의 시력상태를 기준으로 평가한다.'),
    metadata={'source_doc': {'total_pages': 160},
 'doc': {'doc_type': 'terms',
         'file_name': 'meritz_2_petpermint_cat_family_45_82_83_206.pdf',
         'insurer_code': 'meritz',
         'product_code': '2',
         'product_name': '무배당 펫퍼민트 Cat&Family보험 다이렉트2601',
         'total_pages': 1,
         'page': 1},
 'term_type': 'unknown',
 'clause': {'clause_type': 'other', 'risk_domains': ['digestive', 'eye']},
 'indexing': {'chunk_id': 'chunk_000518',
              'chunk_char_len': 284,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'normal',
              'tag_method': 'rule',
              'tag_confidence': 0.25}},
)
