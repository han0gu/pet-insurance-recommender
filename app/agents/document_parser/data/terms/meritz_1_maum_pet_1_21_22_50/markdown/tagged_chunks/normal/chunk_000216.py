from langchain_core.documents import Document

chunk = Document(
    page_content=('종료됨에 따라 제1조(특별약관의 적용범위) 제1항 제2호에서 정한 조건을 만족하지 않\n'
 '게 된 경우에는 이 조항이 적용되지 않습니다.# 【예시】2019년 1월 15일에 전환대상계약에 가입한 계약자가 2019년 6월 1일에 이 '
 '특별약관을\n'
 '청약하고 회사가 승낙하여 전환대상계약이 장애인전용보험으로 전환되었으나 2019년\n'
 '12월 1일에 전환을 취소한 경우, 이 전환대상계약에 납입된 모든 보험료는 당해 연도\n'
 '보험료 납입영수증에 장애인전용 보장성 보험료로 표시되지 않으며 소득세법에 따라 보'),
    metadata={'source_doc': {'total_pages': 50},
 'doc': {'doc_type': 'terms',
         'file_name': 'meritz_1_maum_pet_1_21_22_50.pdf',
         'insurer_code': 'meritz',
         'product_code': '1',
         'product_name': '메리츠 마음든든 반려동물보험',
         'total_pages': 1,
         'page': 1},
 'term_type': 'unknown',
 'clause': {'clause_type': 'coverage', 'risk_domains': ['digestive']},
 'indexing': {'chunk_id': 'chunk_000216',
              'chunk_char_len': 269,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'normal',
              'tag_method': 'rule',
              'tag_confidence': 0.55}},
)
