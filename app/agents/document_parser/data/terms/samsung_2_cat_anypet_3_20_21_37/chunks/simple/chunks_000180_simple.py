from langchain_core.documents import Document

chunk = Document(
    page_content=('제2항에도 불구하고, 「 전환대상계약이 장애인전용보험으로 전환된 당해년도에 제4조(전환 취소)에 따라 전환을 취소하는 경우」 에는 '
 '당해년도에 납입한 모든 전환대상계약보험료가 보험료 납입영수 증에 장애인전용 보장성보험료로 표시되지 않습니다. 다만, '
 '제2조(제출서류)제1항에 따라 제출된 장애인증명서상 장애예상기간(또는 장애기간)이 종료됨에 따라 제1조(특약의 적용범위) 제1항 제2 '
 '호에서 정한 조건을 만족하지 않게 된 경우에는 이 조항이 적용되지 않습니다.\n'
 '【예시】'),
    metadata={'source_doc': {'total_pages': 35},
 'doc': {'doc_type': 'terms',
         'file_name': 'samsung_2_cat_anypet_3_20_21_37.pdf',
         'insurer_code': 'samsung',
         'product_code': '2',
         'product_name': '(일반)반려묘보험 애니펫',
         'total_pages': 35,
         'page': 37},
 'term_type': 'special',
 'clause': {'clause_type': 'coverage', 'risk_domains': ['digestive']},
 'indexing': {'chunk_id': 'chunk_000180',
              'chunk_char_len': 260,
              'embedding_model': 'solar-embedding-1-large',
              'tag_method': 'rule',
              'tag_confidence': 0.8}},
)
