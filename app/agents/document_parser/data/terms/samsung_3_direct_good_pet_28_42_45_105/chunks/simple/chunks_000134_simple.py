from langchain_core.documents import Document

chunk = Document(
    page_content=('서 보험계약대출 원금과 이자를 차감할 수 있습니다.\n'
 '③ 제2항의 규정에도 불구하고 회사는 제27조(보험료의 납입이 연체되는 경우 납입최고 (독촉)와 계약의 해지)에 따라 계약이 해지되는 '
 '때에는 즉시 해약환급금에서 보험계약 대출 원금과 이자를 차감합니다. ④ 회사는 보험수익자에게 보험계약대출 사실을 통지할 수 있습니다.\n'
 '제35조 (배당금의 지급)\n'
 '회사는 이 보험에 대하여 계약자에게 배당금을 지급하지 않습니다.\n'
 '제7관 분쟁의 조정 등\n'
 '제36조 (분쟁의 조정)'),
    metadata={'source_doc': {'total_pages': 78},
 'doc': {'doc_type': 'terms',
         'file_name': 'samsung_3_direct_good_pet_28_42_45_105.pdf',
         'insurer_code': 'samsung',
         'product_code': '3',
         'product_name': '(장기)무배당 삼성화재 다이렉트 '
                         '착한펫보험(강아지)',
         'total_pages': 66,
         'page': 41},
 'term_type': 'basic',
 'clause': {'clause_type': 'exclusion', 'risk_domains': ['other']},
 'indexing': {'chunk_id': 'chunk_000134',
              'chunk_char_len': 256,
              'embedding_model': 'solar-embedding-1-large',
              'tag_method': 'rule',
              'tag_confidence': 0.85}},
)
