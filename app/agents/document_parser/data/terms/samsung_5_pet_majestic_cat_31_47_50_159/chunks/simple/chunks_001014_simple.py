from langchain_core.documents import Document

chunk = Document(
    page_content=('창상봉합술(안면과경부이외,변연절제포함,근육,길이2.5cm 미만) | SC031\n'
 '창상봉합술(안면과경부이외,변연절제포함,근육,길이2.5cm이상~5.0cm미만) | SC032\n'
 '창상봉합술(안면과경부이외,변연절제포함,근육,길이5.0cm이상~10.0cm미만) | SC039\n'
 '창상봉합술(안면과경부이외,변연절제포함,근육,길이10cm이상, 10cm마다 추가) | SC040'),
    metadata={'source_doc': {'total_pages': 129},
 'doc': {'doc_type': 'terms',
         'file_name': 'samsung_5_pet_majestic_cat_31_47_50_159.pdf',
         'insurer_code': 'samsung',
         'product_code': '5',
         'product_name': '(장기)무배당 삼성화재 펫보험 의기냥냥',
         'total_pages': 107,
         'page': 154},
 'term_type': 'special',
 'clause': {'clause_type': 'coverage', 'risk_domains': ['skin', 'joint']},
 'indexing': {'chunk_id': 'chunk_001014',
              'chunk_char_len': 199,
              'embedding_model': 'solar-embedding-1-large',
              'tag_method': 'llm',
              'tag_confidence': 0.95}},
)
