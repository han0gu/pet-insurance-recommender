from langchain_core.documents import Document

chunk = Document(
    page_content=('위험이 증가된 후에 적용해야 할 보험요율(이하「변경후 요\n'
 '율」이라 합니다)에 대한 비율에 따라 보험금을 삭감하여\n'
 '지급합니다. 다만, 증가된 위험과 관계없이 발생한 보험금96지급사유에 관해서는 이를 원래대로 지급합니다.\uf000 계약자 또는 '
 '피보험자가 고의 또는 중대한 과실로 제1항\n'
 '각 호의 변경사실을 회사에 알리지 않았을 경우 변경후 요\n'
 '율이 변경전 요율보다 높을 때에는 회사는 그 변경사실을\n'
 '안 날부터 1개월 이내에 계약자 또는 피보험자에게 제4항에'),
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
 'indexing': {'chunk_id': 'chunk_000175',
              'chunk_char_len': 249,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'simple',
              'tag_method': 'rule',
              'tag_confidence': 0.8}},
)
