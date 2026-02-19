from langchain_core.documents import Document

chunk = Document(
    page_content=('① 계약자가 갱신계약의 제1회 보험료를 납입기일까지 납입하지 않은 때에는 보통약관 제30조(보험료의 납입이 연체되는 경우 '
 '납입최고(독촉)와 계약의 해지)에 따라 납입최 고(독촉)하며, 이 납입최고(독촉)기간 안에 보험료가 납입되지 않은 경우 납입최고(독 '
 '촉)기간이 끝나는 날의 다음날 갱신 계약을 해제합니다. ② 회사는 납입최고(독촉)기간 안에 발생한 사고에 대하여 약정한 보험금을 '
 '지급합니다. 이 경우 계약자는 즉시 갱신계약 보험료를 납입하여야 합니다'),
    metadata={'source_doc': {'total_pages': 107},
 'doc': {'doc_type': 'terms',
         'file_name': 'samsung_4_pet_majestic_dog_33_49_52_139.pdf',
         'insurer_code': 'samsung',
         'product_code': '4',
         'product_name': '(장기)무배당 삼성화재 펫보험 위풍댕댕',
         'total_pages': 91,
         'page': 130},
 'term_type': 'special',
 'clause': {'clause_type': 'renewal', 'risk_domains': ['other']},
 'indexing': {'chunk_id': 'chunk_000814',
              'chunk_char_len': 253,
              'embedding_model': 'solar-embedding-1-large',
              'tag_method': 'rule',
              'tag_confidence': 0.55}},
)
