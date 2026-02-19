from langchain_core.documents import Document

chunk = Document(
    page_content=('3) 한손의 첫째 손가락 이외의 손가락을 잃었을 때 (손가락 하나마다) | 10\n'
 '4) 한손의 5개손가락 모두의 손가락뼈 일부를 잃었을 때 또는 뚜렷한 장해를 남긴 때 | 30\n'
 '5) 한손의 첫째 손가락의 손가락뼈 일부를 잃었을 | 10'),
    metadata={'source_doc': {'total_pages': 160},
 'doc': {'doc_type': 'terms',
         'file_name': 'meritz_2_petpermint_cat_family_45_82_83_206.pdf',
         'insurer_code': 'meritz',
         'product_code': '2',
         'product_name': '무배당 펫퍼민트 Puppy&Family보험 다이렉트2601',
         'total_pages': 160,
         'page': 195},
 'term_type': 'special',
 'clause': {'clause_type': 'coverage', 'risk_domains': ['other', 'joint']},
 'indexing': {'chunk_id': 'chunk_000709',
              'chunk_char_len': 130,
              'embedding_model': 'solar-embedding-1-large',
              'tag_method': 'llm',
              'tag_confidence': 0.95}},
)
