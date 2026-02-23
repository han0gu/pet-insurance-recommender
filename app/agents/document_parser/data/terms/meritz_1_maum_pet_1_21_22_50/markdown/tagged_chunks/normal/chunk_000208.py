from langchain_core.documents import Document

chunk = Document(
    page_content=('- 이 있다고 인정하는 사람\n'
 '【소득세법 시행규칙 제54조(장애아동의 범위)】영 제107조 제1항 제1호에서 "기획재정부령으로 정하는 사람"이란「장애아동 복지지원\n'
 '법」제21조 제1항에 따른 발달재활서비스를 지원받고 있는 사람을 말한다.【이 특별약관을 적용할 수 없는 사례 예시】1. 전환대상계약의 '
 '피보험자 1인은 비장애인이고 보험수익자 2인 중 한명은 비장애인,'),
    metadata={'source_doc': {'total_pages': 50},
 'doc': {'doc_type': 'terms',
         'file_name': 'meritz_1_maum_pet_1_21_22_50.pdf',
         'insurer_code': 'meritz',
         'product_code': '1',
         'product_name': '메리츠 마음든든 반려동물보험',
         'total_pages': 1,
         'page': 1},
 'term_type': 'unknown',
 'clause': {'clause_type': 'other', 'risk_domains': ['digestive']},
 'indexing': {'chunk_id': 'chunk_000208',
              'chunk_char_len': 201,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'normal',
              'tag_method': 'rule',
              'tag_confidence': 0.25}},
)
