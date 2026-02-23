from langchain_core.documents import Document

chunk = Document(
    page_content=('호에서 정한 조건을 만족하지 않게 된 경우에는 이 조항이 적용되지 않습니다.# 【예시】2019년 1월 15일에 전환대상계약에 가입한 '
 '계약자가 2019년 6월 1일에 이 특별약관을 청약하고 회사\n'
 '가 승낙하여 전환대상계약이 장애인전용보험으로 전환되었으나 2019년 12월 1일에 전환을 취소한 경\n'
 '우, 이 전환대상계약에 납입된 모든 보험료는 해당 연도 보험료 납입영수증에 장애인전용 보장성 보험\n'
 '료로 표시되지 않으며 소득세법에 따라 보험료의 100분의 15에 해당하는 금액이 종합소득산출세액에'),
    metadata={'source_doc': {'total_pages': 45},
 'doc': {'doc_type': 'terms',
         'file_name': 'samsung_1_dog_anypet_3_20_21_47.pdf',
         'insurer_code': 'samsung',
         'product_code': '1',
         'product_name': '(일반)반려견보험 애니펫',
         'total_pages': 1,
         'page': 1},
 'term_type': 'unknown',
 'clause': {'clause_type': 'coverage', 'risk_domains': ['digestive']},
 'indexing': {'chunk_id': 'chunk_000184',
              'chunk_char_len': 273,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'normal',
              'tag_method': 'rule',
              'tag_confidence': 0.55}},
)
