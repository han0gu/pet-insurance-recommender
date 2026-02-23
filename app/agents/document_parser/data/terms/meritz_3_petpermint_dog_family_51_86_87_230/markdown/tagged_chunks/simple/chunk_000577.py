from langchain_core.documents import Document

chunk = Document(
    page_content=('| QCA001 | 귀 가려움증 (원인 불명) |  |  |\n'
 '| QFA001 | 발진 (원인 불명) |  |  |\n'
 '| QFA002 | 피부염 (원인 불명) |  |  |\n'
 '| QFA003 | 피부의 가려움증 (원인 불명) |  |  |\n'
 '| QFA004 | 탈모 (원인 불명) |  |  |\n'
 '198Ⅳ. 별표# 【별표1】보험금을 지급할 때의 적립이율 계산\n'
 '(제8조 제5항, 제10조 제3항 및 제35조 제2항 관련)| 구 분 | 기 간 | 지 급 이 자 |\n'
 '| --- | --- | --- |'),
    metadata={'source_doc': {'total_pages': 180},
 'doc': {'doc_type': 'terms',
         'file_name': 'meritz_3_petpermint_dog_family_51_86_87_230.pdf',
         'insurer_code': 'meritz',
         'product_code': '3',
         'product_name': '무배당 펫퍼민트 Puppy&Family보험 다이렉트2601',
         'total_pages': 1,
         'page': 1},
 'term_type': 'unknown',
 'clause': {'clause_type': 'other', 'risk_domains': ['skin']},
 'indexing': {'chunk_id': 'chunk_000577',
              'chunk_char_len': 272,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'simple',
              'tag_method': 'rule',
              'tag_confidence': 0.25}},
)
