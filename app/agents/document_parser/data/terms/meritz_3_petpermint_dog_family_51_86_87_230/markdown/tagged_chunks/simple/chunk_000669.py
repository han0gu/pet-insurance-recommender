from langchain_core.documents import Document

chunk = Document(
    page_content=('| --- | --- |\n'
 '| 1) 한발의 리스프랑관절 이상을 잃었을 때 | 40 |\n'
 '| 2) 한발의 5개발가락을 모두 잃었을 때 | 30 |\n'
 '| 3) 한발의 첫째발가락을 잃었을 때 | 10 |\n'
 '| 4) 한발의 첫째발가락 이외의 발가락을 잃었을 때 (발가락 하나마다) | 5 |\n'
 '| 5) 한발의 5개발가락 모두의 발가락뼈 일부를 잃었을 때 또는 뚜렷한 장해를 남긴 때 | 20 |\n'
 '| 6) 한발의 첫째발가락의 발가락뼈 일부를 잃었을 때 또는 뚜렷한 장해를 남긴 때 | 8 |'),
    metadata={'source_doc': {'total_pages': 180},
 'doc': {'doc_type': 'terms',
         'file_name': 'meritz_3_petpermint_dog_family_51_86_87_230.pdf',
         'insurer_code': 'meritz',
         'product_code': '3',
         'product_name': '무배당 펫퍼민트 Puppy&Family보험 다이렉트2601',
         'total_pages': 1,
         'page': 1},
 'term_type': 'unknown',
 'clause': {'clause_type': 'other', 'risk_domains': ['digestive', 'joint']},
 'indexing': {'chunk_id': 'chunk_000669',
              'chunk_char_len': 264,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'simple',
              'tag_method': 'rule',
              'tag_confidence': 0.25}},
)
