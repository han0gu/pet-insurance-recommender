from langchain_core.documents import Document

chunk = Document(
    page_content=('제34조(회사의 파산선고와 해지)\n'
 '\uf000 회사가 파산의 선고를 받은 때에는 계약자는 계약을 해 지할 수 있습니다. \uf000 제1항의 규정에 따라 해지하지 않은 '
 '계약은 파산선고 후 3개월이 지난 때에는 그 효력을 잃습니다. \uf000 제1항의 규정에 따라 계약이 해지되거나 제2항의 규정에 따라 '
 '계약이 효력을 잃는 경우에 회사는 제35조(해약환급 금) 제1항에 따른 해약환급금을 계약자에게 지급합니다.\n'
 '제35조(해약환급금)'),
    metadata={'source_doc': {'total_pages': 180},
 'doc': {'doc_type': 'terms',
         'file_name': 'meritz_3_petpermint_dog_family_51_86_87_230.pdf',
         'insurer_code': 'meritz',
         'product_code': '3',
         'product_name': '무배당 펫퍼민트 Puppy&Family보험 다이렉트2601',
         'total_pages': 180,
         'page': 81},
 'term_type': 'basic',
 'clause': {'clause_type': 'other', 'risk_domains': ['other']},
 'indexing': {'chunk_id': 'chunk_000151',
              'chunk_char_len': 226,
              'embedding_model': 'solar-embedding-1-large',
              'tag_method': 'llm',
              'tag_confidence': 0.95}},
)
