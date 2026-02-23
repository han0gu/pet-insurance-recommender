from langchain_core.documents import Document

chunk = Document(
    page_content=('됩니다.\n'
 '\uf000 제2항에도 불구하고, "전환대상계약이 장애인전용보험으로 전환된 해당 연도에 제\n'
 '4조(전환 취소)에 따라 전환을 취소하는 경우"에는 해당 연도에 납입한 모든 전환\n'
 '대상계약보험료가 보험료 납입영수증에 장애인전용 보장성보험료로 표시되지 않습\n'
 '니다. 단, 제2조(제출서류)제1항에 따라 제출된 장애인증명서상 장애예상기간(또\n'
 '는 장애기간)이 종료됨에 따라 제1조(적용범위) 제1항 제2호에서 정한 조건을 만족\n'
 '하지 않게 된 경우에는 이 조항이 적용되지 않습니다.| 예 시 | 특별세액공제 대상 기간 예시 |\n'
 '| --- | --- |'),
    metadata={'source_doc': {'total_pages': 113},
 'doc': {'doc_type': 'terms',
         'file_name': 'kb_17390_gold_pet_dog_54_72_74_166.pdf',
         'insurer_code': 'kb',
         'product_code': '17390',
         'product_name': '[일반보험] KB반려행복펫보험',
         'total_pages': 1,
         'page': 1},
 'term_type': 'unknown',
 'clause': {'clause_type': 'coverage', 'risk_domains': ['digestive']},
 'indexing': {'chunk_id': 'chunk_000814',
              'chunk_char_len': 299,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'normal',
              'tag_method': 'rule',
              'tag_confidence': 0.55}},
)
