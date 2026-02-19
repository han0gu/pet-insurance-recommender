from langchain_core.documents import Document

chunk = Document(
    page_content=('분류항목 | 수가코드\n'
 '안면과 경부 이외 | 창상봉합술(안면과경부이외,단순봉합,표재성,길이 5.0cm 이상~10.0cm 미만) | SB029\n'
 '창상봉합술(안면과경부이외,단순봉합,표재성,길이10cm이상, 10cm마다 추가) | SB030\n'
 '창상봉합술(안면과경부이외,단순봉합,근육,길이2.5cm미만) | SB031\n'
 '창상봉합술(안면과경부이외,단순봉합,근육,길이2.5cm이상~5.0cm미만) | SB032\n'
 '창상봉합술(안면과경부이외,단순봉합,근육,길이5.0cm이상~10.0cm미만) | SB039'),
    metadata={'source_doc': {'total_pages': 129},
 'doc': {'doc_type': 'terms',
         'file_name': 'samsung_5_pet_majestic_cat_31_47_50_159.pdf',
         'insurer_code': 'samsung',
         'product_code': '5',
         'product_name': '(장기)무배당 삼성화재 펫보험 의기냥냥',
         'total_pages': 107,
         'page': 154},
 'term_type': 'special',
 'clause': {'clause_type': 'other', 'risk_domains': ['skin', 'head']},
 'indexing': {'chunk_id': 'chunk_001012',
              'chunk_char_len': 269,
              'embedding_model': 'solar-embedding-1-large',
              'tag_method': 'llm',
              'tag_confidence': 0.95}},
)
