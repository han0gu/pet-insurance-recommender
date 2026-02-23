from langchain_core.documents import Document

chunk = Document(
    page_content=('이 특별약관에서 정하지 않은 사항은 보통약관 제1절 일반조항을 따릅니다. 다만,\n'
 '이 특별약관에서는 보통약관 제1절 일반조항 제9조(만기환급금의 지급), 제24조(계# 약의 소멸) 및 제36조(중도인출)는 '
 '제외합니다.3. 6대호흡계특정질환진단비# 제1조(보험금의지급사유)회사는 피보험자가 이 특별약관의 보험기간 중에 6대호흡계특정질환으로 '
 '진단확정된\n'
 '경우에는 아래에 정한 금액을 최초 1회에 한하여 6대호흡계특정질환진단비로 보험수\n'
 '익자에게 지급합니다.| 구 분 | 지 급 금 액 | 지 급 금 액 |\n'
 '| --- | --- | --- |'),
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
 'indexing': {'chunk_id': 'chunk_000390',
              'chunk_char_len': 297,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'simple',
              'tag_method': 'rule',
              'tag_confidence': 0.85}},
)
