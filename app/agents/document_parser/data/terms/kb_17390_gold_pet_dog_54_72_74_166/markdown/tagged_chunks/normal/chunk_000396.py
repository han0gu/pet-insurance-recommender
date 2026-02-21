from langchain_core.documents import Document

chunk = Document(
    page_content=('- 약관의 계약자적립액 및 미경과보험료를 계약자에게 지급합니다.\n'
 '# 제5조(준용규정)질이 특별약관에서 정하지 않은 사항은 보통약관 제1절 일반조항을 따릅니다. 다만,\n'
 '이 특별약관에서는 보통약관 제1절 일반조항 제9조(만기환급금의 지급), 제24조(계\n'
 '약의 소멸) 및 제36조(중도인출)는 제외합니다.\n'
 '상병해4.천식지속상태(급성중증천식)진단비및제1조(보험금의 지급사유)질회사는 피보험자가 이 특별약관의병반\n'
 '에는 이 특별약관의 보험가입금액을 최초 1회에 한하여 천식지속상태(급성중증천식)\n'
 '려\n'
 '진단비로 보험수익자에게 지급합니다. 동'),
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
 'indexing': {'chunk_id': 'chunk_000396',
              'chunk_char_len': 294,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'normal',
              'tag_method': 'rule',
              'tag_confidence': 0.85}},
)
