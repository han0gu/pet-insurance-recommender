from langchain_core.documents import Document

chunk = Document(
    page_content=('서 공제되지 않습니다.전환대상계약에 이 특약이 부가된 이후 제4조(전환 취소)에 따라 전환을 취소한 경우 또는 전환대\n'
 '상계약이 제1조(특약의 적용범위)제1항 제2호에서 정한 조건을 만족하지 않아 이 특약의 효력이\n'
 '없어진 경우 해당 전환대상계약에는 이 특약을 다시 부가할 수 없습니다. 다만, 제2조(제출서류)\n'
 '제1항에 따라 제출된 장애인증명서상 장애예상기간(또는 장애기간)이 종료됨에 따라 전환대상계약\n'
 '이 제1조(특약의 적용범위) 제1항 제2호에서 정한 조건을 만족하지 않게 된 경우에는 이 조항이'),
    metadata={'source_doc': {'total_pages': 45},
 'doc': {'doc_type': 'terms',
         'file_name': 'samsung_1_dog_anypet_3_20_21_47.pdf',
         'insurer_code': 'samsung',
         'product_code': '1',
         'product_name': '(일반)반려견보험 애니펫',
         'total_pages': 1,
         'page': 1},
 'term_type': 'unknown',
 'clause': {'clause_type': 'other', 'risk_domains': ['digestive']},
 'indexing': {'chunk_id': 'chunk_000185',
              'chunk_char_len': 278,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'simple',
              'tag_method': 'rule',
              'tag_confidence': 0.25}},
)
