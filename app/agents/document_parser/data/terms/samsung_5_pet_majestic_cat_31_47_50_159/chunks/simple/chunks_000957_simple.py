from langchain_core.documents import Document

chunk = Document(
    page_content=('7) 한 발의 첫째 발가락 이외의 발가락의 발가락뼈 일부를 잃었을 때 또는 뚜렷 한 장해를 남긴 때(발가락 하나마다) | 3\n'
 '나. 장해판정기준'),
    metadata={'source_doc': {'total_pages': 129},
 'doc': {'doc_type': 'terms',
         'file_name': 'samsung_5_pet_majestic_cat_31_47_50_159.pdf',
         'insurer_code': 'samsung',
         'product_code': '5',
         'product_name': '(장기)무배당 삼성화재 펫보험 의기냥냥',
         'total_pages': 107,
         'page': 146},
 'term_type': 'special',
 'clause': {'clause_type': 'coverage', 'risk_domains': ['joint', 'other']},
 'indexing': {'chunk_id': 'chunk_000957',
              'chunk_char_len': 79,
              'embedding_model': 'solar-embedding-1-large',
              'tag_method': 'llm',
              'tag_confidence': 0.9}},
)
