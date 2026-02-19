from langchain_core.documents import Document

chunk = Document(
    page_content=('② 계약자는 제1항에 따른 보험계약대출금과 그 이자를 언제든지 상환할 수 있으며 상환 하지 않은 때에는 회사는 보험금, 해약환급금 등의 '
 '지급사유가 발생한 날에 지급금에 서 보험계약대출 원금과 이자를 차감할 수 있습니다. ③ 제2항의 규정에도 불구하고 회사는 '
 '제30조(보험료의 납입이 연체되는 경우 납입최고 (독촉)와 계약의 해지)에 따라 계약이 해지되는 때에는 즉시 해약환급금에서 보험계약 대출 '
 '원금과 이자를 차감합니다. ④ 회사는 보험수익자에게 보험계약대출 사실을 통지할 수 있습니다.\n'
 '제38조 (배당금의 지급)'),
    metadata={'source_doc': {'total_pages': 107},
 'doc': {'doc_type': 'terms',
         'file_name': 'samsung_4_pet_majestic_dog_33_49_52_139.pdf',
         'insurer_code': 'samsung',
         'product_code': '4',
         'product_name': '(장기)무배당 삼성화재 펫보험 위풍댕댕',
         'total_pages': 91,
         'page': 48},
 'term_type': 'basic',
 'clause': {'clause_type': 'coverage', 'risk_domains': ['other']},
 'indexing': {'chunk_id': 'chunk_000159',
              'chunk_char_len': 287,
              'embedding_model': 'solar-embedding-1-large',
              'tag_method': 'rule',
              'tag_confidence': 0.8}},
)
