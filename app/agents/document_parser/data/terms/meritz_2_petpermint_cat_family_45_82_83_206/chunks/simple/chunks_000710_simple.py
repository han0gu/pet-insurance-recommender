from langchain_core.documents import Document

chunk = Document(
    page_content=('장해의 분류 | 지급률\n'
 '때 또는 뚜렷한 장해를 남긴 때 6) 한손의 첫째 손가락 이외의 손가락의 손가락뼈 일부를 잃었을 때 또는 뚜렷한 장해를 남긴 때 (손가락 '
 '하나마다) | 5\n'
 '나. 장해판정기준'),
    metadata={'source_doc': {'total_pages': 160},
 'doc': {'doc_type': 'terms',
         'file_name': 'meritz_2_petpermint_cat_family_45_82_83_206.pdf',
         'insurer_code': 'meritz',
         'product_code': '2',
         'product_name': '무배당 펫퍼민트 Puppy&Family보험 다이렉트2601',
         'total_pages': 160,
         'page': 196},
 'term_type': 'special',
 'clause': {'clause_type': 'coverage', 'risk_domains': ['joint', 'other']},
 'indexing': {'chunk_id': 'chunk_000710',
              'chunk_char_len': 109,
              'embedding_model': 'solar-embedding-1-large',
              'tag_method': 'llm',
              'tag_confidence': 0.95}},
)
