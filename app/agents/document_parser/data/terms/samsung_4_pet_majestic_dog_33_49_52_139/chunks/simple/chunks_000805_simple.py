from langchain_core.documents import Document

chunk = Document(
    page_content=('제9조 (준용규정)\n'
 '이 특별약관에 정하지 않은 사항은 특별약관 일반사항을 따릅니다. 특별약관 일반사항에 서도 정하지 않은 사항은 보통약관을 따릅니다. 다만, '
 '보통약관 제5조(보험금을 지급하지 않는 사유), 제10조(환급금의 중도인출), 제11조(만기환급금의 지급)은 제외합니다.'),
    metadata={'source_doc': {'total_pages': 107},
 'doc': {'doc_type': 'terms',
         'file_name': 'samsung_4_pet_majestic_dog_33_49_52_139.pdf',
         'insurer_code': 'samsung',
         'product_code': '4',
         'product_name': '(장기)무배당 삼성화재 펫보험 위풍댕댕',
         'total_pages': 91,
         'page': 127},
 'term_type': 'special',
 'clause': {'clause_type': 'exclusion', 'risk_domains': ['other']},
 'indexing': {'chunk_id': 'chunk_000805',
              'chunk_char_len': 155,
              'embedding_model': 'solar-embedding-1-large',
              'tag_method': 'rule',
              'tag_confidence': 0.85}},
)
