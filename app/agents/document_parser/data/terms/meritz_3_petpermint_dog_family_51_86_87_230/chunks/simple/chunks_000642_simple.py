from langchain_core.documents import Document

chunk = Document(
    page_content=('제4조(계약자의 임의해지)\n'
 '계약자는 계약이 소멸하기 전에는 언제든지 계약을 해지할 수 있으며, 이 경우 회사는 해약환급금을 계약자에게 지급 합니다. 다만, 타인을 '
 '위한 계약의 경우에는 계약자는 그 타인의 동의를 얻거나 보험증권을 소지한 경우에 한하여 계 약을 해지할 수 있습니다.\n'
 '제5조(준용규정)'),
    metadata={'source_doc': {'total_pages': 180},
 'doc': {'doc_type': 'terms',
         'file_name': 'meritz_3_petpermint_dog_family_51_86_87_230.pdf',
         'insurer_code': 'meritz',
         'product_code': '3',
         'product_name': '무배당 펫퍼민트 Puppy&Family보험 다이렉트2601',
         'total_pages': 180,
         'page': 188},
 'term_type': 'special',
 'clause': {'clause_type': 'other', 'risk_domains': ['other']},
 'indexing': {'chunk_id': 'chunk_000642',
              'chunk_char_len': 166,
              'embedding_model': 'solar-embedding-1-large',
              'tag_method': 'llm',
              'tag_confidence': 0.95}},
)
