from langchain_core.documents import Document

chunk = Document(
    page_content=('| 5 | AFA004 | 피지종 |  |\n'
 '| 5 | AFA005 | 모낭상피종 |  |\n'
 '| 5 | AFA006 AFA007 | 기저세포종 비만세포종 (피부) (양성) |  |\n'
 '| 5 | AFB007 | 비만세포종 (피부) (악성) |  |\n'
 '| 5 | AFC007 | 비만세포종(피부) (양성 또는 악성이 불확실 한) |  |\n'
 '| 5 | AFA008 | 흑색종 (양성) |  |\n'
 '| 5 | AFB008 | 흑색종 (악성) |  |\n'
 '| 5 | AFC008 | 흑색종 (양성 또는 악성이 불확실한) |  |'),
    metadata={'source_doc': {'total_pages': 160},
 'doc': {'doc_type': 'terms',
         'file_name': 'meritz_2_petpermint_cat_family_45_82_83_206.pdf',
         'insurer_code': 'meritz',
         'product_code': '2',
         'product_name': '무배당 펫퍼민트 Cat&Family보험 다이렉트2601',
         'total_pages': 1,
         'page': 1},
 'term_type': 'unknown',
 'clause': {'clause_type': 'other', 'risk_domains': ['skin']},
 'indexing': {'chunk_id': 'chunk_000488',
              'chunk_char_len': 281,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'simple',
              'tag_method': 'rule',
              'tag_confidence': 0.25}},
)
