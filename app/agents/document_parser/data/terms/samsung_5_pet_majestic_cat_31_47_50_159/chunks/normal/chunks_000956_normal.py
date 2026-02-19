from langchain_core.documents import Document

chunk = Document(
    page_content=('11. 발가락의 장해\n'
 '가. 장해의 분류\n'
 '장 해 의 분 류 | 지급률(%)\n'
 '1) 한 발의 리스프랑관절 이상을 잃었을 때 | 40\n'
 '2) 한 발의 5개 발가락을 모두 잃었을 때 | 30\n'
 '3) 한 발의 첫째 발가락을 잃었을 때 | 10\n'
 '4) 한 발의 첫째 발가락 이외의 발가락을 잃었을 때(발가락 하나마다) | 5\n'
 '5) 한 발의 5개 발가락 모두의 발가락뼈 일부를 잃었을 때 또는 뚜렷한 장해를 남긴 때 | 20\n'
 '6) 한 발의 첫째 발가락의 발가락뼈 일부를 잃었을 때 또는 뚜렷한 장해를 남긴 때 | 8'),
    metadata={'source_doc': {'total_pages': 129},
 'doc': {'doc_type': 'terms',
         'file_name': 'samsung_5_pet_majestic_cat_31_47_50_159.pdf',
         'insurer_code': 'samsung',
         'product_code': '5',
         'product_name': '(장기)무배당 삼성화재 펫보험 의기냥냥',
         'total_pages': 107,
         'page': 146},
 'term_type': 'special',
 'clause': {'clause_type': 'coverage', 'risk_domains': ['joint']},
 'indexing': {'chunk_id': 'chunk_000956',
              'chunk_char_len': 277,
              'embedding_model': 'solar-embedding-1-large',
              'tag_method': 'llm',
              'tag_confidence': 0.95}},
)
