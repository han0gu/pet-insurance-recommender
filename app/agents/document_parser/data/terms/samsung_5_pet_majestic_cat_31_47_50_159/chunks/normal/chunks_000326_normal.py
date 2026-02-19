from langchain_core.documents import Document

chunk = Document(
    page_content=('. ③ 제2항의 규정에도 불구하고 회사는 제29조(보험료의 납입이 연체되는 경우 납입최고 (독촉)와 특별약관의 해지)에 따라 계약이 '
 '해지되는 때에는 즉시 해약환급금에서 보험 계약대출 원금과 이자를 차감합니다. ④ 회사는 보험수익자에게 보험계약대출 사실을 통지할 수 '
 '있습니다.'),
    metadata={'source_doc': {'total_pages': 129},
 'doc': {'doc_type': 'terms',
         'file_name': 'samsung_5_pet_majestic_cat_31_47_50_159.pdf',
         'insurer_code': 'samsung',
         'product_code': '5',
         'product_name': '(장기)무배당 삼성화재 펫보험 의기냥냥',
         'total_pages': 107,
         'page': 63},
 'term_type': 'special',
 'clause': {'clause_type': 'other', 'risk_domains': ['other']},
 'indexing': {'chunk_id': 'chunk_000326',
              'chunk_char_len': 153,
              'embedding_model': 'solar-embedding-1-large',
              'tag_method': 'llm',
              'tag_confidence': 0.95}},
)
