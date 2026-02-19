from langchain_core.documents import Document

chunk = Document(
    page_content=('니다. \uf000 손해가 제1항 제1호 또는 제2호에 해당되는 사실로 생긴 것이 아님을 계약자 또는 피보험자가 증명한 경우에는 제4 '
 '항에 관계없이 보상합니다. \uf000 회사는 다른 보험가입내역에 대한 계약 전․후 알릴 의무 위반을 이유로 계약을 해지하거나 보험금 '
 '지급을 거절하지 않습니다'),
    metadata={'source_doc': {'total_pages': 180},
 'doc': {'doc_type': 'terms',
         'file_name': 'meritz_3_petpermint_dog_family_51_86_87_230.pdf',
         'insurer_code': 'meritz',
         'product_code': '3',
         'product_name': '무배당 펫퍼민트 Puppy&Family보험 다이렉트2601',
         'total_pages': 180,
         'page': 183},
 'term_type': 'special',
 'clause': {'clause_type': 'coverage', 'risk_domains': ['digestive']},
 'indexing': {'chunk_id': 'chunk_000619',
              'chunk_char_len': 153,
              'embedding_model': 'solar-embedding-1-large',
              'tag_method': 'rule',
              'tag_confidence': 0.8}},
)
