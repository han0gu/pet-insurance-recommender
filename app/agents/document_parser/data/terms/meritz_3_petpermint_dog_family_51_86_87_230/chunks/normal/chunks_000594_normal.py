from langchain_core.documents import Document

chunk = Document(
    page_content=('【가지급보험금】\n'
 '보험금이 지급기한 내에 지급되지 못할 것으로 판단되는 경우 회사가 예상되는 보험금의 일부를 먼저 지급하는 제도로 피보험자가 필요로 하는 '
 '비용을 보전해 주기 위 해 회사가 먼저 지급하는 임시 교부금을 말합니다.\n'
 '제6조(의무보험과의 관계)\n'
 '\uf000 회사는 이 특별약관에 의하여 보상하여야 하는 금액이 의무보험에서 보상하는 금액을 초과할 때에 한하여 그 초과 액만을 '
 '보상합니다. 다만, 의무보험이 다수인 경우에는 제7 조(보험금의 분담)를 따릅니다.'),
    metadata={'source_doc': {'total_pages': 180},
 'doc': {'doc_type': 'terms',
         'file_name': 'meritz_3_petpermint_dog_family_51_86_87_230.pdf',
         'insurer_code': 'meritz',
         'product_code': '3',
         'product_name': '무배당 펫퍼민트 Puppy&Family보험 다이렉트2601',
         'total_pages': 180,
         'page': 178},
 'term_type': 'special',
 'clause': {'clause_type': 'coverage', 'risk_domains': ['other']},
 'indexing': {'chunk_id': 'chunk_000594',
              'chunk_char_len': 254,
              'embedding_model': 'solar-embedding-1-large',
              'tag_method': 'llm',
              'tag_confidence': 0.9}},
)
