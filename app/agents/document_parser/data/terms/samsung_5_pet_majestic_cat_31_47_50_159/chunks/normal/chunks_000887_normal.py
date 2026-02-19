from langchain_core.documents import Document

chunk = Document(
    page_content=('3. 코의 장해\n'
 '가. 장해의 분류\n'
 '장 해 의 분 류 | 지급률(%)\n'
 '1) 코의 호흡기능을 완전히 잃었을 때 | 15\n'
 '2) 코의 후각기능을 완전히 잃었을 때 | 5\n'
 '나. 장해판정기준'),
    metadata={'source_doc': {'total_pages': 129},
 'doc': {'doc_type': 'terms',
         'file_name': 'samsung_5_pet_majestic_cat_31_47_50_159.pdf',
         'insurer_code': 'samsung',
         'product_code': '5',
         'product_name': '(장기)무배당 삼성화재 펫보험 의기냥냥',
         'total_pages': 107,
         'page': 138},
 'term_type': 'special',
 'clause': {'clause_type': 'coverage', 'risk_domains': ['head']},
 'indexing': {'chunk_id': 'chunk_000887',
              'chunk_char_len': 100,
              'embedding_model': 'solar-embedding-1-large',
              'tag_method': 'llm',
              'tag_confidence': 0.95}},
)
