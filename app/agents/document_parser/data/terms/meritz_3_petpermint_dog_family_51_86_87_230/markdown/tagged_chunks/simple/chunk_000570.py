from langchain_core.documents import Document

chunk = Document(
    page_content=('| QGA004 | 핍뇨 (원인 불명) |  |  |\n'
 '| 5 | 피부질환 | AFA001 | 지방종 |\n'
 '| 5 | 피부질환 | AFA002 | 조직구종 (피부) |\n'
 '| 5 | 피부질환 | AFA003 | 유두종 (피부) |\n'
 '| 5 | 피부질환 | AFA004 | 피지종 |\n'
 '| 5 | 피부질환 | AFA005 | 모낭상피종 |\n'
 '| 5 | 피부질환 | AFA006 | 기저세포종 |\n'
 '| 5 | 피부질환 | AFA007 | 비만세포종 (피부) (양성) |'),
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
 'indexing': {'chunk_id': 'chunk_000570',
              'chunk_char_len': 252,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'simple',
              'tag_method': 'rule',
              'tag_confidence': 0.25}},
)
