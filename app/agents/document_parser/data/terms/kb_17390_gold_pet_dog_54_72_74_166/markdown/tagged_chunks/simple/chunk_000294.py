from langchain_core.documents import Document

chunk = Document(
    page_content=('이 특별약관에서는 보통약관 제1절 일반조항 제9조(만기환급금의 지급), 제24조(계\n'
 '약의 소멸) 및 제36조(중도인출)는 제외합니다.7.질외모특정상해(머리, 목)수술비병# 제1조(보험금의 지급사유)회사는 피보험자가 이 '
 '특별약관의 보험기간 중에 상해의 직접결과로써 "외모특정상\n'
 '및\n'
 '해"로 진단확정되고 그 치료를 직접적인 목적으로 수술을 받은 경우 보험가입금액을\n'
 '질\n'
 '외모특정상해(머리,목)수술비로 보험수익자에게 매 사고시마다 지급합니다.'),
    metadata={'source_doc': {'total_pages': 113},
 'doc': {'doc_type': 'terms',
         'file_name': 'kb_17390_gold_pet_dog_54_72_74_166.pdf',
         'insurer_code': 'kb',
         'product_code': '17390',
         'product_name': '[일반보험] KB반려행복펫보험',
         'total_pages': 1,
         'page': 1},
 'term_type': 'unknown',
 'clause': {'clause_type': 'exclusion', 'risk_domains': ['head']},
 'indexing': {'chunk_id': 'chunk_000294',
              'chunk_char_len': 242,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'simple',
              'tag_method': 'rule',
              'tag_confidence': 0.85}},
)
