from langchain_core.documents import Document

chunk = Document(
    page_content=('별\n'
 '이 특별약관에서 정하지 않은 사항은 보통약관 제1절 일반조항을 따릅니다. 다만,\n'
 '약\n'
 '이 특별약관에서는 보통약관 제1절 일반조항 제9조(만기환급금의 지급), 제24조(계# 약의 소멸) 및 제36조(중도인출)는 '
 '제외합니다.9. 골절진단비Ⅱ(치아파절제외)제1조(보험금의 지급사유)\n'
 '회사는 피보험자가 이 특별약관의 보험기간 중에 상해의 직접결과로써 【별표4】(골\n'
 '절분류표Ⅱ(치아파절제외))에서 정한 골절(치아의 파절(깨짐, 부러짐) 제외)로 진\n'
 '단확정 된 경우 이 특별약관의 보험가입금액을 골절진단비로 보험수익자에게 매 사'),
    metadata={'source_doc': {'total_pages': 113},
 'doc': {'doc_type': 'terms',
         'file_name': 'kb_17390_gold_pet_dog_54_72_74_166.pdf',
         'insurer_code': 'kb',
         'product_code': '17390',
         'product_name': '[일반보험] KB반려행복펫보험',
         'total_pages': 1,
         'page': 1},
 'term_type': 'unknown',
 'clause': {'clause_type': 'exclusion', 'risk_domains': ['dental']},
 'indexing': {'chunk_id': 'chunk_000314',
              'chunk_char_len': 290,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'normal',
              'tag_method': 'rule',
              'tag_confidence': 0.85}},
)
