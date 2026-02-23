from langchain_core.documents import Document

chunk = Document(
    page_content=('대출납입이 종료되었음을 서면, 전화(음성녹음) 또는 전자\n'
 '문서(SMS 포함) 등으로 계약자에게 안내하여 드립니다.# 【자동대출납입】보험료를 제때에 납입하기 곤란한 경우에 계약자가 자동대\n'
 '출납입을 신청하면 해당 보험 상품의 해약환급금 범위 내\n'
 '에서 납입할 보험료를 자동적으로 대출하여 이를 보험료\n'
 '납입에 충당하는 서비스를 말합니다.제29조(보험료의 납입이 연체되는 경우 납입최고(독촉)와\n'
 '계약의 해지)\n'
 '\uf000 계약자가 제2회 이후의 보험료를 납입기일까지 납입하지\n'
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
 'clause': {'clause_type': 'other', 'risk_domains': ['digestive']},
 'indexing': {'chunk_id': 'chunk_000102',
              'chunk_char_len': 288,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'simple',
              'tag_method': 'rule',
              'tag_confidence': 0.25}},
)
