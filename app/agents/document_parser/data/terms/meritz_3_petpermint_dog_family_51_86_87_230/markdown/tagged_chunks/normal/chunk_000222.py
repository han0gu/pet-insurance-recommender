from langchain_core.documents import Document

chunk = Document(
    page_content=('지 않은 사항은 보통약관을 따릅니다. 다만, 보통약관 제3\n'
 '조(보험금의 지급사유), 제4조(보험금 지급에 관한 세부규\n'
 '정), 제5조(보험금을 지급하지 않는 사유), 제9조(적립부분\n'
 '적립이율에 관한 사항), 제10조(만기환급금의 지급), 제25\n'
 '조(계약의 소멸) 및 제38조(중도인출)은 제외합니다.1071. 펫퍼민트 반려견 통원의료비보장 특별약관# 제1조(보험금의 지급사유)# ① '
 '고급형\uf000 회사는 보험기간 중에 보험증권에 기재된 반려동물에게\n'
 '질병 또는 상해가 발생하여 그 치료를 직접적인 목적으로'),
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
 'indexing': {'chunk_id': 'chunk_000222',
              'chunk_char_len': 277,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'normal',
              'tag_method': 'rule',
              'tag_confidence': 0.85}},
)
