from langchain_core.documents import Document

chunk = Document(
    page_content=('지급하여야 할 해약환급금이 있을 때에는 제35조(해약환급 금) 제1항에 따른 해약환급금을 계약자에게 지급합니다.\n'
 '제32조의1(위법계약의 해지)'),
    metadata={'source_doc': {'total_pages': 180},
 'doc': {'doc_type': 'terms',
         'file_name': 'meritz_3_petpermint_dog_family_51_86_87_230.pdf',
         'insurer_code': 'meritz',
         'product_code': '3',
         'product_name': '무배당 펫퍼민트 Puppy&Family보험 다이렉트2601',
         'total_pages': 180,
         'page': 80},
 'term_type': 'basic',
 'clause': {'clause_type': 'other', 'risk_domains': ['other']},
 'indexing': {'chunk_id': 'chunk_000145',
              'chunk_char_len': 79,
              'embedding_model': 'solar-embedding-1-large',
              'tag_method': 'llm',
              'tag_confidence': 0.95}},
)
