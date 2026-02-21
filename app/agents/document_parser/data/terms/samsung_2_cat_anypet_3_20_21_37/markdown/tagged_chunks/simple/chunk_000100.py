from langchain_core.documents import Document

chunk = Document(
    page_content=('- 를 해당 보험기간 및 분할회수에 따라 아래에 정한 시기까지 납입하여야 합니다.\n'
 '| 보험 기간 | 제2회 이후 분납보험료 납입시기 | 제2회 이후 분납보험료 납입시기 |\n'
 '| --- | --- | --- |\n'
 '| 보험 기간 | 분할회수 | 이후 분납보험료 제2회 |\n'
 '| 1년 | 2회 | 제1회 분납보험료를 납입한 날로부터 6개월 경과시점의 보험증권에 기재된 납입기 일 안에 분납보험료를 납입 |\n'
 '| 1년 | 4회 | 제1회 분납보험료를 납입한 날로부터 3개월 마다 보험증권에 기재된 납입기일 안에 분납보험료를 납입 |'),
    metadata={'source_doc': {'total_pages': 35},
 'doc': {'doc_type': 'terms',
         'file_name': 'samsung_2_cat_anypet_3_20_21_37.pdf',
         'insurer_code': 'samsung',
         'product_code': '2',
         'product_name': '(일반)반려묘보험 애니펫',
         'total_pages': 1,
         'page': 1},
 'term_type': 'unknown',
 'clause': {'clause_type': 'other', 'risk_domains': ['other']},
 'indexing': {'chunk_id': 'chunk_000100',
              'chunk_char_len': 289,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'simple',
              'tag_method': 'rule',
              'tag_confidence': 0.25}},
)
