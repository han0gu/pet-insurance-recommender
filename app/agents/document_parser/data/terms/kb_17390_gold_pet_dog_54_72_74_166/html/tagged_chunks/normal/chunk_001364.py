from langchain_core.documents import Document

chunk = Document(
    page_content=('. 다만, 계약자의 책임있는 사유로 보험료의 납입이 불가능한 경우에는 거래은행의 지정계좌로부터 제1회 보 험료가 이체된 날을 청약일 및 '
 '제1회 보험료 납입일로 합니다. \uf000 제1항의 경우에 회사는 청약서를 접수한 날로부터 30일 이내에 승낙 또는 거절하 여야 하며, '
 '승낙한 때에는 금융기관의 해당계좌에서 제1회 보험료를 받고 보험증 권을 드립니다. 제2조(계약 후 알릴 의무) 계약자는 지정계좌의 번호가 '
 '변경 또는 거래정지된 경우에는 이 사실을 즉시 회사에 알려야 합니다'),
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
 'indexing': {'chunk_id': 'chunk_001364',
              'chunk_char_len': 262,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'normal',
              'tag_method': 'rule',
              'tag_confidence': 0.55}},
)
