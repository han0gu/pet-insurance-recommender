from langchain_core.documents import Document

chunk = Document(
    page_content=('AFB010 | 편평세포암종\n'
 'AFA011 | 항문주위선종\n'
 'AFB012 | 항문주위선암종\n'
 'AFA013 | 상세미상의 피부 신생물 (양성)\n'
 'AFB013 | 상세미상의 피부 신생물 (악성)\n'
 'AFC013 | 상세미상의 피부 신생물 (양성 또는 악성 이 불확실한)\n'
 'AFA014 | 기타 피부 신생물 (양성)\n'
 'AFB014 | 기타 피부 신생물 (악성)\n'
 'AFC014 | 기타 피부 신생물 (양성 또는 악성이 불 확실한)\n'
 'GAA001 | 외이도염 (세균성)'),
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
 'indexing': {'chunk_id': 'chunk_000684',
              'chunk_char_len': 247,
              'embedding_model': 'solar-embedding-1-large',
              'tag_method': 'llm',
              'tag_confidence': 0.95}},
)
