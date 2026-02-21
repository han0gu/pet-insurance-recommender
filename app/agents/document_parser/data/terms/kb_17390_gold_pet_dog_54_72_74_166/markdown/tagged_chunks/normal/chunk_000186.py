from langchain_core.documents import Document

chunk = Document(
    page_content=('- 환하지 않은 때에는 회사는 보험금, 해약환급금 등의 지급사유가 발생한 날에 지급\n'
 '- 금에서 보험계약대출의 원금과 이자를 차감할 수 있습니다.\n'
 '- \uf000 제2항의 규정에도 불구하고 회사는 제28조(보험료의 납입이 연체되는 경우 납입최\n'
 '- 고(독촉)와 계약의 해지)에 따라 계약이 해지되는 때에는 즉시 해약환급금에서 보\n'
 '- 험계약대출의 원금과 이자를 차감합니다.\n'
 '- \uf000 회사는 보험수익자에게 보험계약대출 사실을 통지할 수 있습니다.'),
    metadata={'source_doc': {'total_pages': 113},
 'doc': {'doc_type': 'terms',
         'file_name': 'kb_17390_gold_pet_dog_54_72_74_166.pdf',
         'insurer_code': 'kb',
         'product_code': '17390',
         'product_name': '[일반보험] KB반려행복펫보험',
         'total_pages': 1,
         'page': 1},
 'term_type': 'unknown',
 'clause': {'clause_type': 'coverage', 'risk_domains': ['other']},
 'indexing': {'chunk_id': 'chunk_000186',
              'chunk_char_len': 239,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'normal',
              'tag_method': 'rule',
              'tag_confidence': 0.55}},
)
