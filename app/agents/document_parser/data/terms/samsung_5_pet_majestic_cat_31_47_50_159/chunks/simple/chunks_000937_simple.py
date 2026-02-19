from langchain_core.documents import Document

chunk = Document(
    page_content=('장 해 의 분 류 | 지급률(%)\n'
 '4) 한 다리의 3대 관절 중 관절 하나의 기능에 심한 장해를 남긴 때 | 20\n'
 '5) 한 다리의 3대 관절 중 관절 하나의 기능에 뚜렷한 장해를 남긴 때 | 10\n'
 '6) 한 다리의 3대 관절 중 관절 하나의 기능에 약간의 장해를 남긴 때 | 5\n'
 '7) 한 다리에 가관절이 남아 뚜렷한 장해를 남긴 때 | 20\n'
 '8) 한 다리에 가관절이 남아 약간의 장해를 남긴 때 | 10\n'
 '9) 한 다리의 뼈에 기형을 남긴 때 | 5\n'
 '10) 한 다리가 5cm 이상 짧아지거나 길어진 때 | 30'),
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
 'indexing': {'chunk_id': 'chunk_000937',
              'chunk_char_len': 283,
              'embedding_model': 'solar-embedding-1-large',
              'tag_method': 'llm',
              'tag_confidence': 0.95}},
)
