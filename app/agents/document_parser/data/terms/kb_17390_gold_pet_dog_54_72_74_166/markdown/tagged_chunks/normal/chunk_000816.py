from langchain_core.documents import Document

chunk = Document(
    page_content=('. | 2022년 1월 15일에 전환대상계약에 가입한 계약자가 2022년 6월 1일에 이 특별 약관을 청약하고 회사가 승낙하여 '
 '전환대상계약이 장애인전용보험으로 전환되 었으나 2022년 12월 1일에 전환을 취소한 경우, 이 전환대상계약에 납입된 모든 보험료는 해당 '
 '연도 보험료 납입영수증에 장애인전용 보장성 보험료로 표시되지 않으며 소득세법에 따라 보험료의 100분의 15에 해당하는 금액이 '
 '종합소득산출 세액에서 공제되지 않습니다. |'),
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
 'indexing': {'chunk_id': 'chunk_000816',
              'chunk_char_len': 243,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'normal',
              'tag_method': 'rule',
              'tag_confidence': 0.55}},
)
