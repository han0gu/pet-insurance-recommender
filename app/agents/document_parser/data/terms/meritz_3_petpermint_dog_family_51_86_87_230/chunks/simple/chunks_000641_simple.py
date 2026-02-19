from langchain_core.documents import Document

chunk = Document(
    page_content=('제3조(특별약관의 소멸)\n'
 '이 특별약관에서 정한 보상하는 손해가 더 이상 발생할 수 없는 경우에는 이 특별약관은 그 때부터 소멸되며, 이 경우 회사는「보험료 및 '
 '해약환급금 산출방법서」에서 정한 이 특별약관의 그 때까지 적립한 계약자적립액 및 미경과보험 료를 지급합니다.\n'
 '제4조(계약자의 임의해지)'),
    metadata={'source_doc': {'total_pages': 180},
 'doc': {'doc_type': 'terms',
         'file_name': 'meritz_3_petpermint_dog_family_51_86_87_230.pdf',
         'insurer_code': 'meritz',
         'product_code': '3',
         'product_name': '무배당 펫퍼민트 Puppy&Family보험 다이렉트2601',
         'total_pages': 180,
         'page': 188},
 'term_type': 'special',
 'clause': {'clause_type': 'coverage', 'risk_domains': ['other']},
 'indexing': {'chunk_id': 'chunk_000641',
              'chunk_char_len': 164,
              'embedding_model': 'solar-embedding-1-large',
              'tag_method': 'rule',
              'tag_confidence': 0.8}},
)
