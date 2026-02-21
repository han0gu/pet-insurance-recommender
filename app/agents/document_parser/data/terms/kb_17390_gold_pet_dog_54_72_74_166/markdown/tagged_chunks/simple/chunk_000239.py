from langchain_core.documents import Document

chunk = Document(
    page_content=('이 특별약관에서 정하지 않은 사항은 보통약관 제1절 일반조항을 따릅니다. 다만,\n'
 '이 특별약관에서는 보통약관 제1절 일반조항 제9조(만기환급금의 지급), 제24조(계약의 소멸) 및 제36조(중도인출)는 '
 '제외합니다.일반상해후유장해(3~100%)# 2.제1조(보험금의 지급사유)회사는 피보험자가 이 특별약관의 보험기간 중에 상해로 '
 '장해분류표(【별표1】(장해\n'
 '분류표) 참조. 이하 같습니다)에서 정한 3~100% 장해지급률에 해당하는 장해상태가\n'
 '되었을 때에는 장해분류표에서 정한 장해지급률을 이 특별약관의 보험가입금액에 곱'),
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
 'indexing': {'chunk_id': 'chunk_000239',
              'chunk_char_len': 288,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'simple',
              'tag_method': 'rule',
              'tag_confidence': 0.85}},
)
