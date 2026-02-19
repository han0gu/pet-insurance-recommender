from langchain_core.documents import Document

chunk = Document(
    page_content=('적립 보험료 | 회사가 적립한 금액을 돌려주는데 필요한 보 험료를 말합니다.\n'
 '【보험료】\n'
 '보험료는 계약자가 계약에 따라 회사에게 지급하여야 하 는 요금을 말하며, 보험료는「보장보험료」와「적립보험 료」로 구성되어 있습니다. '
 '또한, 보험료는 보험금 지급을 위한 위험보험료, 회사가 적립한 금액을 돌려주기 위한 적립부분 순보험료 및 회 사의 사업경비를 위한 '
 '부가보험료로 구성됩니다.'),
    metadata={'source_doc': {'total_pages': 180},
 'doc': {'doc_type': 'terms',
         'file_name': 'meritz_3_petpermint_dog_family_51_86_87_230.pdf',
         'insurer_code': 'meritz',
         'product_code': '3',
         'product_name': '무배당 펫퍼민트 Puppy&Family보험 다이렉트2601',
         'total_pages': 180,
         'page': 53},
 'term_type': 'basic',
 'clause': {'clause_type': 'coverage', 'risk_domains': ['digestive']},
 'indexing': {'chunk_id': 'chunk_000011',
              'chunk_char_len': 209,
              'embedding_model': 'solar-embedding-1-large',
              'tag_method': 'rule',
              'tag_confidence': 0.8}},
)
