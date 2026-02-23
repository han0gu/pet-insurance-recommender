from langchain_core.documents import Document

chunk = Document(
    page_content=('| 안면 또는 경부 | 창상봉합술(안면또는경부,변연절제포함,표재성,길이1.5cm미만) | SA021 |\n'
 '| 안면 또는 경부 | 창상봉합술(안면또는경부,변연절제포함,표재성,길이1.5cm 이상~3.0cm 미만) | SA022 |\n'
 '| 안면과 경부 이외 | 창상봉합술(안면과경부이외,단순봉합,표재성,길이2.5cm미만) | SB021 |\n'
 '| 안면과 경부 이외 | 창상봉합술(안면과경부이외,단순봉합,표재성,길이 2.5cm 이상~5.0cm 미만) | SB022 |'),
    metadata={'source_doc': {'total_pages': 129},
 'doc': {'doc_type': 'terms',
         'file_name': 'samsung_5_pet_majestic_cat_31_47_50_159.pdf',
         'insurer_code': 'samsung',
         'product_code': '5',
         'product_name': '(장기)무배당 삼성화재 펫보험 의기냥냥',
         'total_pages': 1,
         'page': 1},
 'term_type': 'unknown',
 'clause': {'clause_type': 'other', 'risk_domains': ['other']},
 'indexing': {'chunk_id': 'chunk_000893',
              'chunk_char_len': 252,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'simple',
              'tag_method': 'rule',
              'tag_confidence': 0.25}},
)
