from langchain_core.documents import Document

chunk = Document(
    page_content=('관련 특별약관 일반조항」에서 정하지 않은 사항은 보통약\n'
 '관을 따릅니다. 다만, 보통약관 제3조(보험금의 지급사유),\n'
 '제4조(보험금 지급에 관한 세부규정), 제5조(보험금을 지급\n'
 '하지 않는 사유), 제9조(적립부분 적립이율에 관한 사항),\n'
 '제10조(만기환급금의 지급), 제25조(계약의 소멸) 및 제38184조(중도인출)은 제외합니다.1851. 갱신형 펫퍼민트 반려견 '
 '배상책임보장 특별약관# 제1조(보상하는 손해)\uf000 회사는 피보험자가 이 특별약관의 보험기간 중에 보험증\n'
 '권에 기재된 반려견의 행위에 기인하는 우연한 사고로 인하'),
    metadata={'source_doc': {'total_pages': 180},
 'doc': {'doc_type': 'terms',
         'file_name': 'meritz_3_petpermint_dog_family_51_86_87_230.pdf',
         'insurer_code': 'meritz',
         'product_code': '3',
         'product_name': '무배당 펫퍼민트 Puppy&Family보험 다이렉트2601',
         'total_pages': 1,
         'page': 1},
 'term_type': 'unknown',
 'clause': {'clause_type': 'exclusion', 'risk_domains': ['digestive']},
 'indexing': {'chunk_id': 'chunk_000521',
              'chunk_char_len': 293,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'simple',
              'tag_method': 'rule',
              'tag_confidence': 0.85}},
)
