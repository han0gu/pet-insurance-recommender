from langchain_core.documents import Document

chunk = Document(
    page_content=('1월 15일에 전환대상계약에 가입한 계약자가 2019년 6월 1일에 이 특별약관을<br>청약하고 회사가 승낙하여 전환대상계약이 '
 '장애인전용보험으로 전환되었으나 2019년<br>12월 1일에 전환을 취소한 경우, 이 전환대상계약에 납입된 모든 보험료는 당해 '
 '연도<br>보험료 납입영수증에 장애인전용 보장성 보험료로 표시되지 않으며 소득세법에 따라 보<br>험료의 100분의 15에 해당하는 '
 "금액이 종합소득산출세액에서 공제되지 않습니다.</p><p id='73' data-category='paragraph'"),
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
 'indexing': {'chunk_id': 'chunk_000386',
              'chunk_char_len': 281,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'simple',
              'tag_method': 'rule',
              'tag_confidence': 0.8}},
)
