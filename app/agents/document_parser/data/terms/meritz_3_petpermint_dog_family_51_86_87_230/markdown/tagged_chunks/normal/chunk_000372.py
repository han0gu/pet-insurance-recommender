from langchain_core.documents import Document

chunk = Document(
    page_content=('치료가 곤란하여 동물병원에 입실하여 수의사의 관리 하에\n'
 '치료에 전념하는 것을 말합니다.# 제5조(특별약관의 소멸)이 특별약관에서 정한 보상하는 손해가 더 이상 발생할 수\n'
 '없는 경우에는 이 특별약관은 그 때부터 소멸되며, 이 경우\n'
 '회사는「보험료 및 해약환급금 산출방법서」에서 정한 이\n'
 '특별약관의 그 때까지 적립한 계약자적립액 및 미경과보험\n'
 '료를 지급합니다.# 제6조(준용규정)이 특별약관에서 정하지 않은 사항은「반려동물 비용손해\n'
 '관련 특별약관 일반조항」을 따르고,「반려동물 비용손해\n'
 '관련 특별약관 일반조항」에서 정하지 않은 사항은 보통약'),
    metadata={'source_doc': {'total_pages': 180},
 'doc': {'doc_type': 'terms',
         'file_name': 'meritz_3_petpermint_dog_family_51_86_87_230.pdf',
         'insurer_code': 'meritz',
         'product_code': '3',
         'product_name': '무배당 펫퍼민트 Puppy&Family보험 다이렉트2601',
         'total_pages': 1,
         'page': 1},
 'term_type': 'unknown',
 'clause': {'clause_type': 'other', 'risk_domains': ['other']},
 'indexing': {'chunk_id': 'chunk_000372',
              'chunk_char_len': 299,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'normal',
              'tag_method': 'rule',
              'tag_confidence': 0.25}},
)
