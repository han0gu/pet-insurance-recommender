from langchain_core.documents import Document

chunk = Document(
    page_content=('| 안면 또는 경부 | 창상봉합술(안면또는경부,단순봉합,근육,길이1.5cm이상~3.0cm미만) | S0032 |\n'
 '| 안면 또는 경부 | 창상봉합술(안면또는경부,단순봉합,근육,길이3.0cm이상~5.0cm미만) | S0037 |\n'
 '| 안면 또는 경부 | 창상봉합술(안면또는경부,단순봉합,근육,길이5.0cm이상~7.5cm미만) | S0038 |\n'
 '| 안면 또는 경부 | 창상봉합술(안면또는경부,단순봉합,근육,길이7.5cm이상~10.0cm미만) | S0039 |'),
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
 'indexing': {'chunk_id': 'chunk_000872',
              'chunk_char_len': 252,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'simple',
              'tag_method': 'rule',
              'tag_confidence': 0.25}},
)
