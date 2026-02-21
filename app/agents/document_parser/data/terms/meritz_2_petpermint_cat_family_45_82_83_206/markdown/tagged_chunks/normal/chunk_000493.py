from langchain_core.documents import Document

chunk = Document(
    page_content=('| LAA013 | 홍반루프스 |  |  |\n'
 '| LAA014 | 천포창 |  |  |\n'
 '| LAA015 | 지간 피부염 |  |  |\n'
 '| LAA016 | 족피부염 |  |  |\n'
 '| LAA017 | 꼬리샘 과증식 |  |  |\n'
 '| LAA018 | 발톱 주위염 |  |  |\n'
 '| LAA019 | 옴진드기 · 개선충 |  |  |\n'
 '| LAA020 LAA022 | 벼룩 / 진드기 등 외부 기생충 질환 기타 피부 질환 |  |  |\n'
 '| QCA001 | 귀 가려움증 (원인 불명) |  |  |'),
    metadata={'source_doc': {'total_pages': 160},
 'doc': {'doc_type': 'terms',
         'file_name': 'meritz_2_petpermint_cat_family_45_82_83_206.pdf',
         'insurer_code': 'meritz',
         'product_code': '2',
         'product_name': '무배당 펫퍼민트 Cat&Family보험 다이렉트2601',
         'total_pages': 1,
         'page': 1},
 'term_type': 'unknown',
 'clause': {'clause_type': 'other', 'risk_domains': ['digestive', 'skin']},
 'indexing': {'chunk_id': 'chunk_000493',
              'chunk_char_len': 270,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'normal',
              'tag_method': 'rule',
              'tag_confidence': 0.25}},
)
