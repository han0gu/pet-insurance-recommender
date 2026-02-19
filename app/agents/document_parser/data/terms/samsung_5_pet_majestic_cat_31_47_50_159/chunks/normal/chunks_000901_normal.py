from langchain_core.documents import Document

chunk = Document(
    page_content=('. 14) 상실된 치아의 크기가 크든지 또는 치간의 간격이나 치아 배열구조 등의 문제 로 사고와 관계없이 새로운 치아가 결손된 경우에는 '
 '사고로 결손된 치아 수에 따라 지급률을 결정한다. 15) 어린이의 유치는 향후에 영구치로 대체되므로 후유장해의 대상이 되지 않으 나, '
 '선천적으로 영구치 결손이 있는 경우에는 유치의 결손을 후유장해로 평가 한다. 16) 가철성 보철물(신체의 일부에 붙였다 떼었다 할 수 '
 '있는 틀니 등)의 파손은 후 유장해의 대상이 되지 않는다.'),
    metadata={'source_doc': {'total_pages': 129},
 'doc': {'doc_type': 'terms',
         'file_name': 'samsung_5_pet_majestic_cat_31_47_50_159.pdf',
         'insurer_code': 'samsung',
         'product_code': '5',
         'product_name': '(장기)무배당 삼성화재 펫보험 의기냥냥',
         'total_pages': 107,
         'page': 140},
 'term_type': 'special',
 'clause': {'clause_type': 'exclusion', 'risk_domains': ['dental']},
 'indexing': {'chunk_id': 'chunk_000901',
              'chunk_char_len': 258,
              'embedding_model': 'solar-embedding-1-large',
              'tag_method': 'llm',
              'tag_confidence': 0.95}},
)
