from langchain_core.documents import Document

chunk = Document(
    page_content=('제3조(특별약관의 부활(효력회복))\n'
 '회사는 이 특별약관의 부활(효력회복) 청약을 받은 경우에 는 보험계약의 부활(효력회복)을 승낙한 경우에 한하여 보 통약관 '
 '제30조(보험료의 납입을 연체하여 해지된 계약의 부'),
    metadata={'source_doc': {'total_pages': 180},
 'doc': {'doc_type': 'terms',
         'file_name': 'meritz_3_petpermint_dog_family_51_86_87_230.pdf',
         'insurer_code': 'meritz',
         'product_code': '3',
         'product_name': '무배당 펫퍼민트 Puppy&Family보험 다이렉트2601',
         'total_pages': 180,
         'page': 193},
 'term_type': 'special',
 'clause': {'clause_type': 'other', 'risk_domains': ['other']},
 'indexing': {'chunk_id': 'chunk_000668',
              'chunk_char_len': 116,
              'embedding_model': 'solar-embedding-1-large',
              'tag_method': 'llm',
              'tag_confidence': 0.95}},
)
