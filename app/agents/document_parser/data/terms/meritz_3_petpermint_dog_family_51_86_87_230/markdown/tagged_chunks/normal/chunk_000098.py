from langchain_core.documents import Document

chunk = Document(
    page_content=('하며, 회사는 계약자가 보험료를 납입한 경우에는 영수증을\n'
 '발행하여 드립니다. 다만, 금융회사(우체국을 포함합니다)를\n'
 '통하여 보험료를 납입한 경우에는 그 금융회사 발행 증빙서\n'
 '류를 영수증으로 대신합니다.# 【납입기일】계약자가 제2회 이후의 보험료를 납입하기로 한 날을 말합\n'
 '니다.75# 제28조(보험료의 자동대출납입)\uf000 계약자는 제29조(보험료의 납입이 연체되는 경우 납입최\n'
 '고(독촉)와 계약의 해지)에 따른 보험료의 납입최고(독촉)\n'
 '기간이 지나기 전까지 회사가 정한 방법에 따라 보험료의'),
    metadata={'source_doc': {'total_pages': 180},
 'doc': {'doc_type': 'terms',
         'file_name': 'meritz_3_petpermint_dog_family_51_86_87_230.pdf',
         'insurer_code': 'meritz',
         'product_code': '3',
         'product_name': '무배당 펫퍼민트 Puppy&Family보험 다이렉트2601',
         'total_pages': 1,
         'page': 1},
 'term_type': 'unknown',
 'clause': {'clause_type': 'other', 'risk_domains': ['other']},
 'indexing': {'chunk_id': 'chunk_000098',
              'chunk_char_len': 272,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'normal',
              'tag_method': 'rule',
              'tag_confidence': 0.25}},
)
