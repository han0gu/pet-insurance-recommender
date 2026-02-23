from langchain_core.documents import Document

chunk = Document(
    page_content=('수술당일제외, 검사비포함)(재가입형) 특별약관에서 정하지 않은 사항은 특별약관 일반사\n'
 '항을 따릅니다. 특별약관 일반사항에서도 정하지 않은 사항은 보통약관을 따릅니다. 다만, 보통약관 제10조(환급금의 중도인출), '
 '제11조(만기환급금의 지급)은 제외합니다.- 119 -4-5. [갱신형] 반려견 배상책임보장 특별약관# 제 1조 (목적)이 특별약관은 '
 '보험계약자(이하「계약자」라 합니다)와 보험회사(이하「회사」라 합니다)\n'
 '사이에 피보험자가 법률상의 배상책임을 부담함으로써 입은 손해에 대한 위험을 보장하'),
    metadata={'source_doc': {'total_pages': 107},
 'doc': {'doc_type': 'terms',
         'file_name': 'samsung_4_pet_majestic_dog_33_49_52_139.pdf',
         'insurer_code': 'samsung',
         'product_code': '4',
         'product_name': '(장기)무배당 삼성화재 펫보험 위풍댕댕',
         'total_pages': 1,
         'page': 1},
 'term_type': 'unknown',
 'clause': {'clause_type': 'exclusion', 'risk_domains': ['digestive']},
 'indexing': {'chunk_id': 'chunk_000632',
              'chunk_char_len': 278,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'normal',
              'tag_method': 'rule',
              'tag_confidence': 0.85}},
)
