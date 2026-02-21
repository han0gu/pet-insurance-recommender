from langchain_core.documents import Document

chunk = Document(
    page_content=('| 분류항목 | 분류항목 | 수가코드 |\n'
 '| --- | --- | --- |\n'
 '| 안면과 경부 이외 | 창상봉합술(안면과경부이외,단순봉합,표재성,길이 5.0cm 이상~10.0cm 미만) | SB029 |\n'
 '| 안면과 경부 이외 | 창상봉합술(안면과경부이외,단순봉합,표재성,길이10cm이상, 10cm마다 추가) | SB030 |\n'
 '| 안면과 경부 이외 | 창상봉합술(안면과경부이외,단순봉합,근육,길이2.5cm미만) | SB031 |'),
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
 'indexing': {'chunk_id': 'chunk_000876',
              'chunk_char_len': 237,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'normal',
              'tag_method': 'rule',
              'tag_confidence': 0.25}},
)
