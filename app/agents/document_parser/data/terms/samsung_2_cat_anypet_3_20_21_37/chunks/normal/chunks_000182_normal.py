from langchain_core.documents import Document

chunk = Document(
    page_content=('전환대상계약에 이 특약이 부가된 이후 제4조(전환 취소)에 따라 전환을 취소한 경우 또는 전환대 상계약이 제1조(특약의 적용범위)제1항 '
 '제2호에서 정한 조건을 만족하지 않아 이 특약의 효력이 없어진 경우 해당 전환대상계약에는 이 특약을 다시 부가할 수 없습니다. 다만, '
 '제2조(제출서류) 제1항에 따라 제출된 장애인증명서상 장애예상기간(또는 장애기간)이 종료됨에 따라 전환대상계약 이 제1조(특약의 '
 '적용범위) 제1항 제2호에서 정한 조건을 만족하지 않게 된 경우에는 이 조항이 적용되지 않습니다.\n'
 '제4조(전환 취소)'),
    metadata={'source_doc': {'total_pages': 35},
 'doc': {'doc_type': 'terms',
         'file_name': 'samsung_2_cat_anypet_3_20_21_37.pdf',
         'insurer_code': 'samsung',
         'product_code': '2',
         'product_name': '(일반)반려묘보험 애니펫',
         'total_pages': 35,
         'page': 37},
 'term_type': 'special',
 'clause': {'clause_type': 'exclusion', 'risk_domains': ['other']},
 'indexing': {'chunk_id': 'chunk_000182',
              'chunk_char_len': 288,
              'embedding_model': 'solar-embedding-1-large',
              'tag_method': 'llm',
              'tag_confidence': 0.95}},
)
