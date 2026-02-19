from langchain_core.documents import Document

chunk = Document(
    page_content=('③ 제2항에도 불구하고,「전환대상계약이 장애인전용보험으로 전환된 당해 연도에 제4조 (전환 취소)에 따라 전환을 취소하는 경우」에는 당해 '
 '연도에 납입한 모든 전환대상계 약보험료가 보험료 납입영수증에 장애인전용 보장성보험료로 표시되지 않습니다. 다만, 제2조(제출서류) '
 '제1항에 따라 제출된 장애인증명서상 장애예상기간(또는 장애기간)이 종료됨에 따라 제1조(특별약관의 적용범위) 제1항 제2호에서 정한 '
 '조건을 만족하지 않 게 된 경우에는 이 조항이 적용되지 않습니다.\n'
 '【예시】'),
    metadata={'source_doc': {'total_pages': 50},
 'doc': {'doc_type': 'terms',
         'file_name': 'meritz_1_maum_pet_1_21_22_50.pdf',
         'insurer_code': 'meritz',
         'product_code': '1',
         'product_name': '메리츠 마음든든 반려동물보험',
         'total_pages': 50,
         'page': 46},
 'term_type': 'special',
 'clause': {'clause_type': 'coverage', 'risk_domains': ['digestive']},
 'indexing': {'chunk_id': 'chunk_000251',
              'chunk_char_len': 265,
              'embedding_model': 'solar-embedding-1-large',
              'tag_method': 'rule',
              'tag_confidence': 0.55}},
)
