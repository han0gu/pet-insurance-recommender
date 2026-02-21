from langchain_core.documents import Document

chunk = Document(
    page_content=('특별약관의 그 때까지 적립한 계약자적립액 및 미경과보험\n'
 '료를 지급합니다.# 제4조(계약자의 임의해지)계약자는 계약이 소멸하기 전에는 언제든지 계약을 해지할\n'
 '수 있으며, 이 경우 회사는 해약환급금을 계약자에게 지급\n'
 '합니다. 다만, 타인을 위한 계약의 경우에는 계약자는 그\n'
 '타인의 동의를 얻거나 보험증권을 소지한 경우에 한하여 계\n'
 '약을 해지할 수 있습니다.# 제5조(준용규정)이 특별약관에서 정하지 않은 사항은「배상책임 관련 특별\n'
 '약관 일반조항」을 따르고,「배상책임 관련 특별약관 일반\n'
 '조항」에서 정하지 않은 사항은「반려동물 비용손해 관련'),
    metadata={'source_doc': {'total_pages': 180},
 'doc': {'doc_type': 'terms',
         'file_name': 'meritz_3_petpermint_dog_family_51_86_87_230.pdf',
         'insurer_code': 'meritz',
         'product_code': '3',
         'product_name': '무배당 펫퍼민트 Puppy&Family보험 다이렉트2601',
         'total_pages': 1,
         'page': 1},
 'term_type': 'unknown',
 'clause': {'clause_type': 'other', 'risk_domains': ['digestive']},
 'indexing': {'chunk_id': 'chunk_000532',
              'chunk_char_len': 299,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'normal',
              'tag_method': 'rule',
              'tag_confidence': 0.25}},
)
