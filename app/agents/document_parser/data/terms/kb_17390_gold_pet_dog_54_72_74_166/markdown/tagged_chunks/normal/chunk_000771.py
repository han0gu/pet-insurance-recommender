from langchain_core.documents import Document

chunk = Document(
    page_content=('| 제1조(보험료의 납입) \uf000 계약자가 제1회 보험료의 납입방법을 계약자의 거래은행 지정 계좌를 통한 자동납 입으로 가입하고자 '
 '하는 경우에, 회사는 청약서를 접수하고 자동이체신청에 필요 한 정보를 제공한 때를 청약일 및 제1회 보험료 납입일로 하여 보통약관 제1절 '
 '일 반조항 제18조(보험계약의 성립)의 규정을 적용합니다. 다만, 계약자의 책임있는 사유로 보험료의 납입이 불가능한 경우에는 거래은행의 '
 '지정계좌로부터 제1회 보 험료가 이체된 날을 청약일 및 제1회 보험료 납입일로 합니다'),
    metadata={'source_doc': {'total_pages': 113},
 'doc': {'doc_type': 'terms',
         'file_name': 'kb_17390_gold_pet_dog_54_72_74_166.pdf',
         'insurer_code': 'kb',
         'product_code': '17390',
         'product_name': '[일반보험] KB반려행복펫보험',
         'total_pages': 1,
         'page': 1},
 'term_type': 'unknown',
 'clause': {'clause_type': 'claim', 'risk_domains': ['other']},
 'indexing': {'chunk_id': 'chunk_000771',
              'chunk_char_len': 271,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'normal',
              'tag_method': 'rule',
              'tag_confidence': 0.55}},
)
