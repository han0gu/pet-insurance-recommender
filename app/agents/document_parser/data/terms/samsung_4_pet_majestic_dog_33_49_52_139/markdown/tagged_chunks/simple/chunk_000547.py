from langchain_core.documents import Document

chunk = Document(
    page_content=('90일 이전에는 계약을 취소 또는 해지할 수 있습니다.)\n'
 '⑩ 제7항 내지 제9항에 따라 계약이 해지된 경우 회사는 특별약관 일반사항 제35조(해약\n'
 '환급금) 제1항에 따른 해약환급금을 계약자에게 지급합니다.- \n'
 '- 108 -# 제28조 (준용규정)이 특별약관에 정하지 않은 사항은 특별약관 일반사항을 따르며, 이 특별약관 및 특별약\n'
 '관 일반사항에 정하지 않은 사항은 보통약관을 따릅니다. 다만, 보통약관 제10조(환급금'),
    metadata={'source_doc': {'total_pages': 107},
 'doc': {'doc_type': 'terms',
         'file_name': 'samsung_4_pet_majestic_dog_33_49_52_139.pdf',
         'insurer_code': 'samsung',
         'product_code': '4',
         'product_name': '(장기)무배당 삼성화재 펫보험 위풍댕댕',
         'total_pages': 1,
         'page': 1},
 'term_type': 'unknown',
 'clause': {'clause_type': 'other', 'risk_domains': ['other']},
 'indexing': {'chunk_id': 'chunk_000547',
              'chunk_char_len': 233,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'simple',
              'tag_method': 'rule',
              'tag_confidence': 0.25}},
)
