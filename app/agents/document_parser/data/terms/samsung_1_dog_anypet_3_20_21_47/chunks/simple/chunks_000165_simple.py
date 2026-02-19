from langchain_core.documents import Document

chunk = Document(
    page_content=('보험 기간 | 제2회 이후 분납보험료 납입시기\n'
 '분할회수 | 이후 분납보험료 제2회\n'
 '1년 | 2회 | 제1회 분납보험료를 납입한 날로부터 6개월 경과시점의 보험증권에 기재된 납입기 일 안에 분납보험료를 납입\n'
 '4회 | 제1회 분납보험료를 납입한 날로부터 3개월 마다 보험증권에 기재된 납입기일 안에 분납보험료를 납입\n'
 '12회 | 제1회 분납보험료를 납입한 날로부터 매1개월 마다 보험증권에 기재된 납입기일 안 에 분납보험료를 납입'),
    metadata={'source_doc': {'total_pages': 45},
 'doc': {'doc_type': 'terms',
         'file_name': 'samsung_1_dog_anypet_3_20_21_47.pdf',
         'insurer_code': 'samsung',
         'product_code': '1',
         'product_name': '(일반)반려견보험 애니펫',
         'total_pages': 45,
         'page': 32},
 'term_type': 'special',
 'clause': {'clause_type': 'other', 'risk_domains': ['other']},
 'indexing': {'chunk_id': 'chunk_000165',
              'chunk_char_len': 237,
              'embedding_model': 'solar-embedding-1-large',
              'tag_method': 'llm',
              'tag_confidence': 0.95}},
)
