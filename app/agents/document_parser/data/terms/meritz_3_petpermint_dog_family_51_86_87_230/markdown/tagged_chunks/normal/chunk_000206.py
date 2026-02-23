from langchain_core.documents import Document

chunk = Document(
    page_content=('은 것으로 봅니다.103\uf000 회사가 청약과 함께 제1회 보험료를 받고 청약을 승낙하\n'
 '기 전에 보험금 지급사유가 발생하였을 때에도 보장개시일\n'
 '부터 이 약관이 정하는 바에 따라 보장을 합니다.【보장개시일】회사가 보장을 개시하는 날로서 계약이 성립되고 제1회 보\n'
 '험료를 받은 날을 말하나, 회사가 승낙하기 전이라도 청약\n'
 '과 함께 제1회 보험료를 받은 경우에는 제1회 보험료를 받\n'
 '은 날을 말합니다. 또한, 보장개시일을 계약일로 봅니다.제17조(보험료의 납입이 연체되는 경우 납입최고(독촉)와'),
    metadata={'source_doc': {'total_pages': 180},
 'doc': {'doc_type': 'terms',
         'file_name': 'meritz_3_petpermint_dog_family_51_86_87_230.pdf',
         'insurer_code': 'meritz',
         'product_code': '3',
         'product_name': '무배당 펫퍼민트 Puppy&Family보험 다이렉트2601',
         'total_pages': 1,
         'page': 1},
 'term_type': 'unknown',
 'clause': {'clause_type': 'coverage', 'risk_domains': ['digestive']},
 'indexing': {'chunk_id': 'chunk_000206',
              'chunk_char_len': 270,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'normal',
              'tag_method': 'rule',
              'tag_confidence': 0.55}},
)
