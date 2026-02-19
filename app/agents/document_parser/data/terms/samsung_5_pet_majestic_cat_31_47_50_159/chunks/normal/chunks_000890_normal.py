from langchain_core.documents import Document

chunk = Document(
    page_content=('4. 씹어먹거나 말하는 장해\n'
 '가. 장해의 분류\n'
 '장 해 의 분 류 | 지급률(%)\n'
 '1) 씹어먹는 기능과 말하는 기능 모두에 심한 장해를 남긴 때 | 100\n'
 '2) 씹어먹는 기능에 심한 장해를 남긴 때 | 80\n'
 '3) 말하는 기능에 심한 장해를 남긴 때 | 60\n'
 '4) 씹어먹는 기능과 말하는 기능 모두에 뚜렷한 장해를 남긴 때 | 40\n'
 '5) 씹어먹는 기능 또는 말하는 기능에 뚜렷한 장해를 남긴 때 | 20\n'
 '6) 씹어먹는 기능과 말하는 기능 모두에 약간의 장해를 남긴때 | 10'),
    metadata={'source_doc': {'total_pages': 129},
 'doc': {'doc_type': 'terms',
         'file_name': 'samsung_5_pet_majestic_cat_31_47_50_159.pdf',
         'insurer_code': 'samsung',
         'product_code': '5',
         'product_name': '(장기)무배당 삼성화재 펫보험 의기냥냥',
         'total_pages': 107,
         'page': 138},
 'term_type': 'special',
 'clause': {'clause_type': 'coverage', 'risk_domains': ['dental', 'head']},
 'indexing': {'chunk_id': 'chunk_000890',
              'chunk_char_len': 263,
              'embedding_model': 'solar-embedding-1-large',
              'tag_method': 'llm',
              'tag_confidence': 0.95}},
)
