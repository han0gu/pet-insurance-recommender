from langchain_core.documents import Document

chunk = Document(
    page_content=('자 또는 피보험자가 최초계약 청약시(2회 이상 부활이 이루 어진 경우 종전 모든 부활 청약 포함) 제7조(계약 전 알릴 의무)를 위반한 '
 '경우에는 제9조(알릴 의무 위반의 효과)가 적용됩니다.\n'
 '제19조(강제집행 등으로 인하여 해지된 계약의 특별부활(효 력회복))'),
    metadata={'source_doc': {'total_pages': 180},
 'doc': {'doc_type': 'terms',
         'file_name': 'meritz_3_petpermint_dog_family_51_86_87_230.pdf',
         'insurer_code': 'meritz',
         'product_code': '3',
         'product_name': '무배당 펫퍼민트 Puppy&Family보험 다이렉트2601',
         'total_pages': 180,
         'page': 106},
 'term_type': 'special',
 'clause': {'clause_type': 'other', 'risk_domains': ['other']},
 'indexing': {'chunk_id': 'chunk_000269',
              'chunk_char_len': 145,
              'embedding_model': 'solar-embedding-1-large',
              'tag_method': 'llm',
              'tag_confidence': 0.95}},
)
