from langchain_core.documents import Document

chunk = Document(
    page_content=('창상봉합술(안면과경부이외,단순봉합,표재성,길이 2.5cm 이상~5.0cm 미만) | SB022\n'
 '창상봉합술(안면과경부이외,변연절제포함,표재성,길이2.5cm미만) | SC021\n'
 '창상봉합술(안면과경부이외,변연절제포함,표재성,길이2.5cm이상~5.0cm미만) | SC022'),
    metadata={'source_doc': {'total_pages': 129},
 'doc': {'doc_type': 'terms',
         'file_name': 'samsung_5_pet_majestic_cat_31_47_50_159.pdf',
         'insurer_code': 'samsung',
         'product_code': '5',
         'product_name': '(장기)무배당 삼성화재 펫보험 의기냥냥',
         'total_pages': 107,
         'page': 157},
 'term_type': 'special',
 'clause': {'clause_type': 'other', 'risk_domains': ['skin', 'head']},
 'indexing': {'chunk_id': 'chunk_001030',
              'chunk_char_len': 148,
              'embedding_model': 'solar-embedding-1-large',
              'tag_method': 'llm',
              'tag_confidence': 0.95}},
)
