from langchain_core.documents import Document

chunk = Document(
    page_content=('3) 한 손의 첫째 손가락 이외의 손가락을 잃었을 때(손가락 하나마다) | 10\n'
 '4) 한 손의 5개 손가락 모두의 손가락뼈 일부를 잃었을 때 또는 뚜렷한 장해를 남긴 때 | 30\n'
 '5) 한 손의 첫째 손가락의 손가락뼈 일부를 잃었을 때 또는 뚜렷한 장해를 남긴 때 | 10\n'
 '6) 한 손의 첫째 손가락 이외의 손가락의 손가락뼈 일부를 잃었을 때 또는 뚜렷 한 장해를 남긴 때(손가락 하나마다) | 5\n'
 '나. 장해판정기준'),
    metadata={'source_doc': {'total_pages': 129},
 'doc': {'doc_type': 'terms',
         'file_name': 'samsung_5_pet_majestic_cat_31_47_50_159.pdf',
         'insurer_code': 'samsung',
         'product_code': '5',
         'product_name': '(장기)무배당 삼성화재 펫보험 의기냥냥',
         'total_pages': 107,
         'page': 145},
 'term_type': 'special',
 'clause': {'clause_type': 'coverage', 'risk_domains': ['joint', 'other']},
 'indexing': {'chunk_id': 'chunk_000949',
              'chunk_char_len': 231,
              'embedding_model': 'solar-embedding-1-large',
              'tag_method': 'llm',
              'tag_confidence': 0.95}},
)
