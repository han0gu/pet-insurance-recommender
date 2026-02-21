from langchain_core.documents import Document

chunk = Document(
    page_content=('이 특별약관에서는 보통약관 제1절 일반조항 제9조(만기환급금의 지급), 제24조(계\n'
 '약의 소멸) 및 제36조(중도인출)는 제외합니다.7. 정신질환특정진단비(연간1회한)이 특별약관의 계약자적립액 등을 지급한 경우에는, 이미 '
 '지급된 계약자적립액# 등을 차감하고 그 차액을 지급합니다.제3조(특정정신질환의 정의 및 진단확정)\n'
 '\uf000 이 특별약관에 있어서 "특정정신질환"이라 함은 제9차 한국표준질병․사인분류에있어서 【별표16】(특정정신질환 분류표)에서 정한 '
 '질병을 말합니다.'),
    metadata={'source_doc': {'total_pages': 113},
 'doc': {'doc_type': 'terms',
         'file_name': 'kb_17390_gold_pet_dog_54_72_74_166.pdf',
         'insurer_code': 'kb',
         'product_code': '17390',
         'product_name': '[일반보험] KB반려행복펫보험',
         'total_pages': 1,
         'page': 1},
 'term_type': 'unknown',
 'clause': {'clause_type': 'exclusion', 'risk_domains': ['other']},
 'indexing': {'chunk_id': 'chunk_000422',
              'chunk_char_len': 260,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'simple',
              'tag_method': 'rule',
              'tag_confidence': 0.85}},
)
