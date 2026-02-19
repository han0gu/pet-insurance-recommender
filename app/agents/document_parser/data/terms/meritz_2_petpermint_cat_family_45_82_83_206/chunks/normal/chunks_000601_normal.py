from langchain_core.documents import Document

chunk = Document(
    page_content=('AFB008 | 흑색종 (악성)\n'
 'AFC008 | 흑색종 (양성 또는 악성이 불확실한)\n'
 'AFB009 | 피부 림프종\n'
 'AFB010 | 편평세포암종\n'
 'AFA011 | 항문주위선종\n'
 'AFB012 | 항문주위선암종\n'
 'AFA013 | 상세미상의 피부 신생물 (양성)\n'
 'AFB013 | 상세미상의 피부 신생물 (악성)\n'
 'AFC013 | 상세미상의 피부 신생물 (양성 또는 악성이 불확실한)\n'
 'AFA014 | 기타 피부 신생물 (양성)\n'
 'AFB014 | 기타 피부 신생물 (악성)\n'
 'AFC014 | 기타 피부 신생물 (양성 또는 악성이 불확실 한)'),
    metadata={'source_doc': {'total_pages': 160},
 'doc': {'doc_type': 'terms',
         'file_name': 'meritz_2_petpermint_cat_family_45_82_83_206.pdf',
         'insurer_code': 'meritz',
         'product_code': '2',
         'product_name': '무배당 펫퍼민트 Puppy&Family보험 다이렉트2601',
         'total_pages': 160,
         'page': 171},
 'term_type': 'special',
 'clause': {'clause_type': 'definition', 'risk_domains': ['skin']},
 'indexing': {'chunk_id': 'chunk_000601',
              'chunk_char_len': 290,
              'embedding_model': 'solar-embedding-1-large',
              'tag_method': 'llm',
              'tag_confidence': 0.95}},
)
