from langchain_core.documents import Document

chunk = Document(
    page_content=('민법 등 관계 법령을 따릅니다.\n'
 '제48조(예금보험에 의한 지급보장)\n'
 '회사가 파산 등으로 인하여 보험금 등을 지급하지 못할 경 우에는 예금자보호법에서 정하는 바에 따라 그 지급을 보장 합니다.\n'
 '【예금자보호제도】\n'
 '예금자보호제도란 예금보험공사가 평소에 금융기관으로 부터 보험료를 받아 기금을 적립한 후, 금융기관이 영업 정지나 파산 등으로 예금을 '
 '지급할 수 없게되면 금융기 관을 대신하여 예금을 지급하는 제도를 말합니다.'),
    metadata={'source_doc': {'total_pages': 160},
 'doc': {'doc_type': 'terms',
         'file_name': 'meritz_2_petpermint_cat_family_45_82_83_206.pdf',
         'insurer_code': 'meritz',
         'product_code': '2',
         'product_name': '무배당 펫퍼민트 Puppy&Family보험 다이렉트2601',
         'total_pages': 160,
         'page': 82},
 'term_type': 'basic',
 'clause': {'clause_type': 'coverage', 'risk_domains': ['digestive']},
 'indexing': {'chunk_id': 'chunk_000179',
              'chunk_char_len': 232,
              'embedding_model': 'solar-embedding-1-large',
              'tag_method': 'rule',
              'tag_confidence': 0.8}},
)
