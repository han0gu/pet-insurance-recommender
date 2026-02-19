from langchain_core.documents import Document

chunk = Document(
    page_content=('\uf000 계약자는 제29조(보험료의 납입이 연체되는 경우 납입최 고(독촉)와 계약의 해지)에 따른 보험료의 납입최고(독촉) 기간이 '
 '지나기 전까지 회사가 정한 방법에 따라 보험료의 자동대출납입을 신청할 수 있으며, 이 경우 제36조(보험계 약대출) 제1항에 따른 '
 '보험계약대출금으로 보험료가 자동으 로 납입되어 계약은 유효하게 지속됩니다. 다만, 계약자가 서면 이외에 인터넷 또는 전화(음성녹음) '
 '등으로 자동대출 납입을 신청할 경우 회사는 자동대출납입 신청내역을 서면 또는 전화(음성녹음) 등으로 계약자에게 알려드립니다'),
    metadata={'source_doc': {'total_pages': 160},
 'doc': {'doc_type': 'terms',
         'file_name': 'meritz_2_petpermint_cat_family_45_82_83_206.pdf',
         'insurer_code': 'meritz',
         'product_code': '2',
         'product_name': '무배당 펫퍼민트 Puppy&Family보험 다이렉트2601',
         'total_pages': 160,
         'page': 72},
 'term_type': 'basic',
 'clause': {'clause_type': 'other', 'risk_domains': ['other']},
 'indexing': {'chunk_id': 'chunk_000124',
              'chunk_char_len': 285,
              'embedding_model': 'solar-embedding-1-large',
              'tag_method': 'llm',
              'tag_confidence': 0.95}},
)
