from langchain_core.documents import Document

chunk = Document(
    page_content=('장 해 의 분 류 | 지급률(%)\n'
 '1) 두 팔의 손목 이상을 잃었을 때 | 100\n'
 '2) 한 팔의 손목 이상을 잃었을 때 | 60\n'
 '3) 한 팔의 3대 관절 중 관절 하나의 기능을 완전히 잃었을 때 | 30\n'
 '4) 한 팔의 3대 관절 중 관절 하나의 기능에 심한 장해를 남긴 때 | 20\n'
 '5) 한 팔의 3대 관절 중 관절 하나의 기능에 뚜렷한 장해를 남긴 때 | 10\n'
 '6) 한 팔의 3대 관절 중 관절 하나의 기능에 약간의 장해를 남긴 때 | 5\n'
 '7) 한 팔에 가관절이 남아 뚜렷한 장해를 남긴 때 | 20'),
    metadata={'source_doc': {'total_pages': 129},
 'doc': {'doc_type': 'terms',
         'file_name': 'samsung_5_pet_majestic_cat_31_47_50_159.pdf',
         'insurer_code': 'samsung',
         'product_code': '5',
         'product_name': '(장기)무배당 삼성화재 펫보험 의기냥냥',
         'total_pages': 107,
         'page': 142},
 'term_type': 'special',
 'clause': {'clause_type': 'coverage', 'risk_domains': ['joint']},
 'indexing': {'chunk_id': 'chunk_000926',
              'chunk_char_len': 280,
              'embedding_model': 'solar-embedding-1-large',
              'tag_method': 'llm',
              'tag_confidence': 0.95}},
)
