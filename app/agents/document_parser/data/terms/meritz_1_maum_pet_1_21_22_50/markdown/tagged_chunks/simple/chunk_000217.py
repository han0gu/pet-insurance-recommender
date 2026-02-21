from langchain_core.documents import Document

chunk = Document(
    page_content=('보험료 납입영수증에 장애인전용 보장성 보험료로 표시되지 않으며 소득세법에 따라 보\n'
 '험료의 100분의 15에 해당하는 금액이 종합소득산출세액에서 공제되지 않습니다.④ 전환대상계약에 이 특별약관이 부가된 이후 제4조(전환 '
 '취소)에 따라 전환을 취소한 경\n'
 '우 또는 전환대상계약이 제1조(특별약관의 적용범위) 제1항 제2호에서 정한 조건을 만\n'
 '족하지 않아 이 특별약관의 효력이 없어진 경우 해당 전환대상계약에는 이 특별약관을- 46 -다시 부가할 수 없습니다. 다만, '
 '제2조(제출서류) 제1항에 따라 제출된 장애인증명서상'),
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
 'indexing': {'chunk_id': 'chunk_000217',
              'chunk_char_len': 287,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'simple',
              'tag_method': 'rule',
              'tag_confidence': 0.8}},
)
