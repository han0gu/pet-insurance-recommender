from langchain_core.documents import Document

chunk = Document(
    page_content=('종류에 따라 보험계약대출이 제한될 수도 있습니다.\n'
 '\uf000 계약자는 제1항에 따른 보험계약대출금과 그 이자를 언\n'
 '제든지 상환할 수 있으며 상환하지 않은 때에는 회사는 보81험금, 해약환급금 등의 지급사유가 발생한 날에 지급금에서\n'
 '보험계약대출의 원금과 이자를 차감할 수 있습니다.\n'
 '\uf000 제2항의 규정에도 불구하고 회사는 제29조(보험료의 납\n'
 '입이 연체되는 경우 납입최고(독촉)와 계약의 해지)에 따라\n'
 '계약이 해지되는 때에는 즉시 해약환급금에서 보험계약대출\n'
 '의 원금과 이자를 차감합니다.\n'
 '\uf000 회사는 보험수익자에게 보험계약대출 사실을 통지할 수'),
    metadata={'source_doc': {'total_pages': 180},
 'doc': {'doc_type': 'terms',
         'file_name': 'meritz_3_petpermint_dog_family_51_86_87_230.pdf',
         'insurer_code': 'meritz',
         'product_code': '3',
         'product_name': '무배당 펫퍼민트 Puppy&Family보험 다이렉트2601',
         'total_pages': 1,
         'page': 1},
 'term_type': 'unknown',
 'clause': {'clause_type': 'coverage', 'risk_domains': ['other']},
 'indexing': {'chunk_id': 'chunk_000124',
              'chunk_char_len': 297,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'simple',
              'tag_method': 'rule',
              'tag_confidence': 0.8}},
)
