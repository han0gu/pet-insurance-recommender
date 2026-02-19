from langchain_core.documents import Document

chunk = Document(
    page_content=('6) 씹어먹는 기능과 말하는 기능 모두에 약간의 장해를 남긴때 | 10\n'
 '7) 씹어먹는 기능 또는 말하는 기능에 약간의 장해를 남긴 때 | 5\n'
 '8) 치아에 14개 이상의 결손이 생긴 때 | 20\n'
 '9) 치아에 7개 이상의 결손이 생긴 때 | 10\n'
 '10) 치아에 5개 이상의 결손이 생긴 때 | 5\n'
 '나. 장해의 평가기준'),
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
 'indexing': {'chunk_id': 'chunk_000891',
              'chunk_char_len': 175,
              'embedding_model': 'solar-embedding-1-large',
              'tag_method': 'llm',
              'tag_confidence': 0.95}},
)
