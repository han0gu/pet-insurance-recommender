from langchain_core.documents import Document

chunk = Document(
    page_content=('① 계약자는 제30조(보험료의 납입이 연체되는 경우 납입최고(독촉)와 계약의 해지)에 따 른 보험료의 납입최고(독촉)기간이 지나기 전까지 '
 '회사가 정한 방법에 따라 보험료의 자동대출납입을 신청할 수 있으며, 이 경우 제37조(보험계약대출) 제1항에 따른 보험 계약대출금으로 '
 '보험료가 자동으로 납입되어 계약은 유효하게 지속됩니다. 다만, 계약 자가 서면 이외에 인터넷 또는 전화(음성녹음) 등으로 자동대출납입을 '
 '신청할 경우 회 사는 자동대출납입 신청내역을 서면, 전화(음성녹음) 또는 전자문서(SMS포함) 등으로 계약자에게 알려 드립니다.'),
    metadata={'source_doc': {'total_pages': 107},
 'doc': {'doc_type': 'terms',
         'file_name': 'samsung_4_pet_majestic_dog_33_49_52_139.pdf',
         'insurer_code': 'samsung',
         'product_code': '4',
         'product_name': '(장기)무배당 삼성화재 펫보험 위풍댕댕',
         'total_pages': 91,
         'page': 44},
 'term_type': 'basic',
 'clause': {'clause_type': 'other', 'risk_domains': ['other']},
 'indexing': {'chunk_id': 'chunk_000125',
              'chunk_char_len': 299,
              'embedding_model': 'solar-embedding-1-large',
              'tag_method': 'llm',
              'tag_confidence': 0.95}},
)
