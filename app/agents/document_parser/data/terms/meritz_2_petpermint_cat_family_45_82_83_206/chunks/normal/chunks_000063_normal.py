from langchain_core.documents import Document

chunk = Document(
    page_content=('\uf000 계약자 또는 피보험자가 고의 또는 중대한 과실로 제1항 각 호의 변경사실을 회사에 알리지 않았을 경우 변경후 요 율이 변경전 '
 '요율보다 높을 때에는 회사는 그 변경사실을 안 날로부터 1개월 이내에 계약자 또는 피보험자에게 제4항 에 의해 보장됨을 통보하고 이에 '
 '따라 보험금을 지급합니 다.'),
    metadata={'source_doc': {'total_pages': 160},
 'doc': {'doc_type': 'terms',
         'file_name': 'meritz_2_petpermint_cat_family_45_82_83_206.pdf',
         'insurer_code': 'meritz',
         'product_code': '2',
         'product_name': '무배당 펫퍼민트 Puppy&Family보험 다이렉트2601',
         'total_pages': 160,
         'page': 60},
 'term_type': 'basic',
 'clause': {'clause_type': 'coverage', 'risk_domains': ['digestive']},
 'indexing': {'chunk_id': 'chunk_000063',
              'chunk_char_len': 162,
              'embedding_model': 'solar-embedding-1-large',
              'tag_method': 'rule',
              'tag_confidence': 0.55}},
)
