from langchain_core.documents import Document

chunk = Document(
    page_content=('9. 다리의 장해\n'
 '가. 장해의 분류\n'
 '장 해 의 분 류 | 지급률(%)\n'
 '1) 두 다리의 발목 이상을 잃었을 때 | 100\n'
 '2) 한 다리의 발목 이상을 잃었을 때 | 60\n'
 '3) 한 다리의 3대 관절 중 관절 하나의 기능을 완전히 잃었을 때 | 30'),
    metadata={'source_doc': {'total_pages': 129},
 'doc': {'doc_type': 'terms',
         'file_name': 'samsung_5_pet_majestic_cat_31_47_50_159.pdf',
         'insurer_code': 'samsung',
         'product_code': '5',
         'product_name': '(장기)무배당 삼성화재 펫보험 의기냥냥',
         'total_pages': 107,
         'page': 143},
 'term_type': 'special',
 'clause': {'clause_type': 'coverage', 'risk_domains': ['joint', 'other']},
 'indexing': {'chunk_id': 'chunk_000936',
              'chunk_char_len': 136,
              'embedding_model': 'solar-embedding-1-large',
              'tag_method': 'llm',
              'tag_confidence': 0.95}},
)
