from langchain_core.documents import Document

chunk = Document(
    page_content=('3) 한발의 첫째발가락을 잃었을 때 | 10\n'
 '4) 한발의 첫째발가락 이외의 발가락을 잃었을 때 (발가락 하나마다) | 5\n'
 '5) 한발의 5개발가락 모두의 발가락뼈 일부를 잃었을 때 또는 뚜렷한 장해를 남긴 때 | 20\n'
 '6) 한발의 첫째발가락의 발가락뼈 일부를 잃었을 때 또는 뚜렷한 장해를 남긴 때 | 8\n'
 '7) 한발의 첫째발가락 이외의 발가락의 발가락 뼈 일부를 잃었을 때 또는 뚜렷한 장해를 남 긴 때(발가락 하나마다) | 3\n'
 '나. 장해판정기준'),
    metadata={'source_doc': {'total_pages': 180},
 'doc': {'doc_type': 'terms',
         'file_name': 'meritz_3_petpermint_dog_family_51_86_87_230.pdf',
         'insurer_code': 'meritz',
         'product_code': '3',
         'product_name': '무배당 펫퍼민트 Puppy&Family보험 다이렉트2601',
         'total_pages': 180,
         'page': 222},
 'term_type': 'special',
 'clause': {'clause_type': 'coverage', 'risk_domains': ['joint', 'other']},
 'indexing': {'chunk_id': 'chunk_000792',
              'chunk_char_len': 248,
              'embedding_model': 'solar-embedding-1-large',
              'tag_method': 'llm',
              'tag_confidence': 0.95}},
)
