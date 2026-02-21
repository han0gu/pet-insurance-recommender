from langchain_core.documents import Document

chunk = Document(
    page_content=('- 약자 또는 피보험자의 책임있는 사유로 보험료의 납입이 불가능한 경우에는 거래은행의 지정계좌\n'
 '- 로부터 제1회 보험료가 이체된 날을 기준으로 합니다)를 청약일 및 제1회 보험료 납입일로 하여\n'
 '- 보통약관의 제15조(보험계약의 성립)과 제21 조(제1회 보험료 등 및 회사의 보장개시)의 규정을 적\n'
 '- 용합니다.\n'
 '- ② 제1항의 경우에 회사는 청약서를 접수한 날로부터 30일 이내에 승낙 또는 거절하여야 하며, 승낙\n'
 '- 한 때에는 지정계좌에서 제1회 보험료를 받고 보험증권을 교부합니다.'),
    metadata={'source_doc': {'total_pages': 35},
 'doc': {'doc_type': 'terms',
         'file_name': 'samsung_2_cat_anypet_3_20_21_37.pdf',
         'insurer_code': 'samsung',
         'product_code': '2',
         'product_name': '(일반)반려묘보험 애니펫',
         'total_pages': 1,
         'page': 1},
 'term_type': 'unknown',
 'clause': {'clause_type': 'coverage', 'risk_domains': ['digestive']},
 'indexing': {'chunk_id': 'chunk_000104',
              'chunk_char_len': 272,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'simple',
              'tag_method': 'rule',
              'tag_confidence': 0.8}},
)
