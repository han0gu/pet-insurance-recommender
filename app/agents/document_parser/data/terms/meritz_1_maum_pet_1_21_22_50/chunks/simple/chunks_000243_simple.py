from langchain_core.documents import Document

chunk = Document(
    page_content=('【소득세법 시행규칙 제54조(장애아동의 범위)】\n'
 '영 제107조 제1항 제1호에서 "기획재정부령으로 정하는 사람"이란「장애아동 복지지원 법」제21조 제1항에 따른 발달재활서비스를 지원받고 '
 '있는 사람을 말한다.\n'
 '【이 특별약관을 적용할 수 없는 사례 예시】\n'
 '1. 전환대상계약의 피보험자 1인은 비장애인이고 보험수익자 2인 중 한명은 비장애인, 한명은 장애인인 경우\n'
 ': 모든 보험수익자가 장애인이 아니므로 이 특별약관을 적용할 수 없습니다.'),
    metadata={'source_doc': {'total_pages': 50},
 'doc': {'doc_type': 'terms',
         'file_name': 'meritz_1_maum_pet_1_21_22_50.pdf',
         'insurer_code': 'meritz',
         'product_code': '1',
         'product_name': '메리츠 마음든든 반려동물보험',
         'total_pages': 50,
         'page': 45},
 'term_type': 'special',
 'clause': {'clause_type': 'exclusion', 'risk_domains': ['other']},
 'indexing': {'chunk_id': 'chunk_000243',
              'chunk_char_len': 242,
              'embedding_model': 'solar-embedding-1-large',
              'tag_method': 'llm',
              'tag_confidence': 0.95}},
)
