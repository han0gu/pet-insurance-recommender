from langchain_core.documents import Document

chunk = Document(
    page_content=('【예시】\n'
 '2019년 1월 15일에 전환대상계약에 가입한 계약자가 2019년 6월 1일에 이 특별약관을 청약하고 회사 가 승낙하여 전환대상계약이 '
 '장애인전용보험으로 전환되었으나 2019년 12월 1일에 전환을 취소한 경 우, 이 전환대상계약에 납입된 모든 보험료는 해당 연도 보험료 '
 '납입영수증에 장애인전용 보장성 보험 료로 표시되지 않으며 소득세법에 따라 보험료의 100분의 15에 해당하는 금액이 종합소득산출세액에 서 '
 '공제되지 않습니다.'),
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
 'indexing': {'chunk_id': 'chunk_000181',
              'chunk_char_len': 243,
              'embedding_model': 'solar-embedding-1-large',
              'tag_method': 'rule',
              'tag_confidence': 0.55}},
)
