from langchain_core.documents import Document

chunk = Document(
    page_content=('- 14) 상실된 치아의 크기가 크든지 또는 치간의 간격이나\n'
 '- 치아 배열구조 등의 문제로 사고와 관계없이 새로\n'
 '- 운 치아가 결손된 경우에는 사고로 결손된 치아 수\n'
 '- 에 따라 지급률을 결정한다.\n'
 '- 15) 어린이의 유치는 향후에 영구치로 대체되므로 후유\n'
 '183# 장해의 대상이 되지 않으나, 선천적으로 영구치 결\n'
 '손이 있는 경우에는 유치의 결손을 후유장해로 평가\n'
 '한다.16) 가철성 보철물(신체의 일부에 붙였다 떼었다 할 수\n'
 '있는 틀니 등)의 파손은 후유장해의 대상이 되지 않'),
    metadata={'source_doc': {'total_pages': 160},
 'doc': {'doc_type': 'terms',
         'file_name': 'meritz_2_petpermint_cat_family_45_82_83_206.pdf',
         'insurer_code': 'meritz',
         'product_code': '2',
         'product_name': '무배당 펫퍼민트 Cat&Family보험 다이렉트2601',
         'total_pages': 1,
         'page': 1},
 'term_type': 'unknown',
 'clause': {'clause_type': 'other', 'risk_domains': ['dental', 'digestive']},
 'indexing': {'chunk_id': 'chunk_000544',
              'chunk_char_len': 268,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'normal',
              'tag_method': 'rule',
              'tag_confidence': 0.25}},
)
