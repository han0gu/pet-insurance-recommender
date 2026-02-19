from langchain_core.documents import Document

chunk = Document(
    page_content=('계약자 또는 피보험자는 제1회 보험료의 납입방법을 거래은행 지정계좌를 통한 자동납입으로 가입 하고자 하는 경우에, 회사는 청약서를 '
 '접수하고 자동이체신청에 필요한 정보를 제공한 때(다만, 계 약자 또는 피보험자의 책임있는 사유로 보험료의 납입이 불가능한 경우에는 '
 '거래은행의 지정계좌 로부터 제1회 보험료가 이체된 날을 기준으로 합니다)를 청약일 및 제1회 보험료 납입일로 하여 보통약관의 '
 '제15조(보험계약의 성립)과 제21 조(제1회 보험료 등 및 회사의 보장개시)의 규정을 적 용합니다'),
    metadata={'source_doc': {'total_pages': 45},
 'doc': {'doc_type': 'terms',
         'file_name': 'samsung_1_dog_anypet_3_20_21_47.pdf',
         'insurer_code': 'samsung',
         'product_code': '1',
         'product_name': '(일반)반려견보험 애니펫',
         'total_pages': 45,
         'page': 34},
 'term_type': 'special',
 'clause': {'clause_type': 'claim', 'risk_domains': ['digestive']},
 'indexing': {'chunk_id': 'chunk_000169',
              'chunk_char_len': 271,
              'embedding_model': 'solar-embedding-1-large',
              'tag_method': 'rule',
              'tag_confidence': 0.55}},
)
