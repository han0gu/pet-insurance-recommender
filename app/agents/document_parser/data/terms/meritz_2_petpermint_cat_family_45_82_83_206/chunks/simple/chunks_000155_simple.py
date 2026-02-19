from langchain_core.documents import Document

chunk = Document(
    page_content=('보험계약대출의 원금과 이자를 차감할 수 있습니다.\n'
 '\uf000 제2항의 규정에도 불구하고 회사는 제29조(보험료의 납 입이 연체되는 경우 납입최고(독촉)와 계약의 해지)에 따라 계약이 '
 '해지되는 때에는 즉시 해약환급금에서 보험계약대출 의 원금과 이자를 차감합니다. \uf000 회사는 보험수익자에게 보험계약대출 사실을 '
 '통지할 수 있습니다.\n'
 '제37조(배당금의 지급)\n'
 '회사는 이 계약에 대하여 계약자에게 배당금을 지급하지 않 습니다.\n'
 '제38조(중도인출)'),
    metadata={'source_doc': {'total_pages': 160},
 'doc': {'doc_type': 'terms',
         'file_name': 'meritz_2_petpermint_cat_family_45_82_83_206.pdf',
         'insurer_code': 'meritz',
         'product_code': '2',
         'product_name': '무배당 펫퍼민트 Puppy&Family보험 다이렉트2601',
         'total_pages': 160,
         'page': 78},
 'term_type': 'basic',
 'clause': {'clause_type': 'exclusion', 'risk_domains': ['other']},
 'indexing': {'chunk_id': 'chunk_000155',
              'chunk_char_len': 240,
              'embedding_model': 'solar-embedding-1-large',
              'tag_method': 'rule',
              'tag_confidence': 0.85}},
)
