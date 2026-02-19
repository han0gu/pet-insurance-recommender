from langchain_core.documents import Document

chunk = Document(
    page_content=('AFC014 | 기타 피부 신생물 (양성 또는 악성이 불확실 한)\n'
 'GAA001 | 외이도염 (세균성)\n'
 'GAA002 GAA003 | 외이도염 (말라세지아) 외이도염 (알러지성)'),
    metadata={'source_doc': {'total_pages': 160},
 'doc': {'doc_type': 'terms',
         'file_name': 'meritz_2_petpermint_cat_family_45_82_83_206.pdf',
         'insurer_code': 'meritz',
         'product_code': '2',
         'product_name': '무배당 펫퍼민트 Puppy&Family보험 다이렉트2601',
         'total_pages': 160,
         'page': 171},
 'term_type': 'special',
 'clause': {'clause_type': 'other', 'risk_domains': ['skin', 'other']},
 'indexing': {'chunk_id': 'chunk_000602',
              'chunk_char_len': 97,
              'embedding_model': 'solar-embedding-1-large',
              'tag_method': 'llm',
              'tag_confidence': 0.95}},
)
