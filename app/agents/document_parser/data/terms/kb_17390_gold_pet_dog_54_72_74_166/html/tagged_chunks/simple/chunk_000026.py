from langchain_core.documents import Document

chunk = Document(
    page_content=('첫 번째 비공휴일까지 대체공휴일로 한다.<br>법<br>③ 제1항 및 제2항에 따른 대체공휴일이 토요일인 경우에는 그 다음의 첫 번째 '
 'ㆍ<br>비공휴일을 대체공휴일로 한다'),
    metadata={'source_doc': {'total_pages': 113},
 'doc': {'doc_type': 'terms',
         'file_name': 'kb_17390_gold_pet_dog_54_72_74_166.pdf',
         'insurer_code': 'kb',
         'product_code': '17390',
         'product_name': '[일반보험] KB반려행복펫보험',
         'total_pages': 1,
         'page': 1},
 'term_type': 'unknown',
 'clause': {'clause_type': 'other', 'risk_domains': ['other']},
 'indexing': {'chunk_id': 'chunk_000026',
              'chunk_char_len': 95,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'simple',
              'tag_method': 'rule',
              'tag_confidence': 0.25}},
)
