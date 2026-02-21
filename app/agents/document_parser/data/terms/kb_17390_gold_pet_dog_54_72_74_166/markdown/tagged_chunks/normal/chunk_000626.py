from langchain_core.documents import Document

chunk = Document(
    page_content=('따릅니다. 다만, 이 특별약관에서는 보통약관 제1절 일반조항 제9조(만기환급금\n'
 '의 지급), 제24조(계약의 소멸) 및 제36조(중도인출)는 제외합니다.- 116 -3. 무지개다리위로금(강아지, '
 '사망)【갱신계약】(【갱신계약】은 자동갱신으로 운영합니다)- 제1조(보험금의 지급사유)\n'
 '- \uf000 회사는 보험증권에 기재된 반려동물이 이 특별약관의 보험기간 중 무지개다리위로\n'
 '- 금의 보장개시일(이하 무지개다리위로금보장개시일이라 합니다) 이후에 사망한\n'
 '- 경우 이 특별약관의 보험가입금액을 무지개다리위로금(강아지, 사망)으로 보험수'),
    metadata={'source_doc': {'total_pages': 113},
 'doc': {'doc_type': 'terms',
         'file_name': 'kb_17390_gold_pet_dog_54_72_74_166.pdf',
         'insurer_code': 'kb',
         'product_code': '17390',
         'product_name': '[일반보험] KB반려행복펫보험',
         'total_pages': 1,
         'page': 1},
 'term_type': 'unknown',
 'clause': {'clause_type': 'exclusion', 'risk_domains': ['digestive']},
 'indexing': {'chunk_id': 'chunk_000626',
              'chunk_char_len': 290,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'normal',
              'tag_method': 'rule',
              'tag_confidence': 0.85}},
)
