from langchain_core.documents import Document

chunk = Document(
    page_content=('분류항목 | 수가코드\n'
 '안면 또는 경부 | 창상봉합술(안면또는경부,단순봉합,표재성,길이 3.0cm 이상~5.0cm 미만) | S0027\n'
 '창상봉합술(안면또는경부,단순봉합,표재성,길이5.0cm이상~7.5cm미만) | S0028\n'
 '창상봉합술(안면또는경부,단순봉합,표재성,길이7.5cm이상~10.0cm미만) | S0029\n'
 '창상봉합술(안면또는경부,단순봉합,표재성,길이10cm이상, 5cm마다 추가) | S0030\n'
 '창상봉합술(안면또는경부,단순봉합,근육,길이1.5cm미만) | S0031'),
    metadata={'source_doc': {'total_pages': 129},
 'doc': {'doc_type': 'terms',
         'file_name': 'samsung_5_pet_majestic_cat_31_47_50_159.pdf',
         'insurer_code': 'samsung',
         'product_code': '5',
         'product_name': '(장기)무배당 삼성화재 펫보험 의기냥냥',
         'total_pages': 107,
         'page': 154},
 'term_type': 'special',
 'clause': {'clause_type': 'coverage', 'risk_domains': ['skin', 'head']},
 'indexing': {'chunk_id': 'chunk_001008',
              'chunk_char_len': 263,
              'embedding_model': 'solar-embedding-1-large',
              'tag_method': 'llm',
              'tag_confidence': 0.95}},
)
