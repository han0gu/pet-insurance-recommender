from langchain_core.documents import Document

chunk = Document(
    page_content=('기 전에 보험금 지급사유가 발생하였을 때에도 보장개시일\n'
 '부터 이 약관이 정하는 바에 따라 보장을 합니다.# 【보장개시일】회사가 보장을 개시하는 날로서 계약이 성립되고 제1회 보\n'
 '험료를 받은 날을 말하나, 회사가 승낙하기 전이라도 청약\n'
 '과 함께 제1회 보험료를 받은 경우에는 제1회 보험료를 받\n'
 '은 날을 말합니다. 또한, 보장개시일을 계약일로 봅니다.제17조(보험료의 납입이 연체되는 경우 납입최고(독촉)와\n'
 '계약의 해지)\uf000 계약자가 제2회 이후의 보험료를 납입기일까지 납입하지\n'
 '않아 보험료 납입이 연체 중인 경우에 회사는 14일(보험기'),
    metadata={'source_doc': {'total_pages': 160},
 'doc': {'doc_type': 'terms',
         'file_name': 'meritz_2_petpermint_cat_family_45_82_83_206.pdf',
         'insurer_code': 'meritz',
         'product_code': '2',
         'product_name': '무배당 펫퍼민트 Cat&Family보험 다이렉트2601',
         'total_pages': 1,
         'page': 1},
 'term_type': 'unknown',
 'clause': {'clause_type': 'coverage', 'risk_domains': ['digestive']},
 'indexing': {'chunk_id': 'chunk_000205',
              'chunk_char_len': 298,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'simple',
              'tag_method': 'rule',
              'tag_confidence': 0.8}},
)
