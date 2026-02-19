from langchain_core.documents import Document

chunk = Document(
    page_content=('QGA003 QGA004 | 비정상 성분의 소변 (원인 불명) 핍뇨 (원인 불명)\n'
 '5 | AFA001 | 지방종\n'
 'AFA002 | 조직구종 (피부)\n'
 'AFA003 | 유두종 (피부)\n'
 'AFA004 | 피지종\n'
 'AFA005 | 모낭상피종\n'
 'AFA006 AFA007 | 기저세포종 비만세포종 (피부) (양성)\n'
 'AFB007 | 비만세포종 (피부) (악성)\n'
 'AFC007 | 비만세포종(피부) (양성 또는 악성이 불확실 한)\n'
 'AFA008 | 흑색종 (양성)\n'
 'AFB008 | 흑색종 (악성)\n'
 'AFC008 | 흑색종 (양성 또는 악성이 불확실한)'),
    metadata={'source_doc': {'total_pages': 160},
 'doc': {'doc_type': 'terms',
         'file_name': 'meritz_2_petpermint_cat_family_45_82_83_206.pdf',
         'insurer_code': 'meritz',
         'product_code': '2',
         'product_name': '무배당 펫퍼민트 Puppy&Family보험 다이렉트2601',
         'total_pages': 160,
         'page': 171},
 'term_type': 'special',
 'clause': {'clause_type': 'exclusion', 'risk_domains': ['urinary', 'skin']},
 'indexing': {'chunk_id': 'chunk_000600',
              'chunk_char_len': 293,
              'embedding_model': 'solar-embedding-1-large',
              'tag_method': 'llm',
              'tag_confidence': 0.95}},
)
