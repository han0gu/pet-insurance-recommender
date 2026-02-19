from langchain_core.documents import Document

chunk = Document(
    page_content=('【 부활(효력회복) 】\n'
 '보험료 납입을 연체하여 계약이 해지되고 계약자가 해약 환급금을 받지 않은 경우 회사가 정하는 소정의 절차에 따라 해지된 계약을 다시 '
 '되살리는 것을 말합니다.'),
    metadata={'source_doc': {'total_pages': 180},
 'doc': {'doc_type': 'terms',
         'file_name': 'meritz_3_petpermint_dog_family_51_86_87_230.pdf',
         'insurer_code': 'meritz',
         'product_code': '3',
         'product_name': '무배당 펫퍼민트 Puppy&Family보험 다이렉트2601',
         'total_pages': 180,
         'page': 105},
 'term_type': 'special',
 'clause': {'clause_type': 'definition', 'risk_domains': ['other']},
 'indexing': {'chunk_id': 'chunk_000267',
              'chunk_char_len': 100,
              'embedding_model': 'solar-embedding-1-large',
              'tag_method': 'rule',
              'tag_confidence': 0.55}},
)
