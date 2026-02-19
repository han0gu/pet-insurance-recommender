from langchain_core.documents import Document

chunk = Document(
    page_content=('① 보험계약자는 제1회 보험료의 납입방법을 보험계약자의 거래은행 지정계좌를 통한 자 동납입으로 가입하고자 하는 경우에, 회사는 청약서를 '
 '접수하고 자동이체신청에 필요한 정보를 제공한 때(다만, 보험계약자의 귀책사유로 보험료의 납입이 불가능한 경우에는 거래은행의 지정계좌로부터 '
 '제1회 보험료가 이체된 날을 기준으로 합니다)를 청약일 및 제1회 보험료 납입일로 하여 보통약관의 제19조(보험계약의 성립)의 규정을 '
 '적용합니 다'),
    metadata={'source_doc': {'total_pages': 50},
 'doc': {'doc_type': 'terms',
         'file_name': 'meritz_1_maum_pet_1_21_22_50.pdf',
         'insurer_code': 'meritz',
         'product_code': '1',
         'product_name': '메리츠 마음든든 반려동물보험',
         'total_pages': 50,
         'page': 36},
 'term_type': 'special',
 'clause': {'clause_type': 'claim', 'risk_domains': ['other']},
 'indexing': {'chunk_id': 'chunk_000200',
              'chunk_char_len': 234,
              'embedding_model': 'solar-embedding-1-large',
              'tag_method': 'rule',
              'tag_confidence': 0.55}},
)
