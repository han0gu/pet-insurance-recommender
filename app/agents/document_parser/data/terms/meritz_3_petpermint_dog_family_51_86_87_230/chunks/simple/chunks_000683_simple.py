from langchain_core.documents import Document

chunk = Document(
    page_content=('AFA003 | 유두종 (피부)\n'
 'AFA004 | 피지종\n'
 'AFA005 | 모낭상피종\n'
 'AFA006 | 기저세포종\n'
 'AFA007 | 비만세포종 (피부) (양성)\n'
 'AFB007 | 악성 비만세포종 (피부) (악성) 비만세포종(피부) (양성 또는 악성이 불 확실한)\n'
 'AFC007 | 흑색종 (양성)\n'
 'AFA008 | 흑색종 (악성)\n'
 'AFB008 AFC008 | 흑색종 (양성 또는 악성이 불확실한)\n'
 'AFB009 | 피부 림프종\n'
 'AFB010 | 편평세포암종\n'
 'AFA011 | 항문주위선종\n'
 'AFB012 | 항문주위선암종'),
    metadata={'source_doc': {'total_pages': 180},
 'doc': {'doc_type': 'terms',
         'file_name': 'meritz_3_petpermint_dog_family_51_86_87_230.pdf',
         'insurer_code': 'meritz',
         'product_code': '3',
         'product_name': '무배당 펫퍼민트 Puppy&Family보험 다이렉트2601',
         'total_pages': 180,
         'page': 197},
 'term_type': 'special',
 'clause': {'clause_type': 'other', 'risk_domains': ['skin']},
 'indexing': {'chunk_id': 'chunk_000683',
              'chunk_char_len': 279,
              'embedding_model': 'solar-embedding-1-large',
              'tag_method': 'llm',
              'tag_confidence': 0.95}},
)
