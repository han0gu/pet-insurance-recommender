from langchain_core.documents import Document

chunk = Document(
    page_content=('. \uf000 제2항에도 불구하고 계약자가 계약내용 변경을 원하지 않거나, 새로운 보장내용으로 계약내용을 변경하는 것이 불 가능한 '
 '경우, 회사는 계약자에게 이 계약의「보험료 및 해 약환급금 산출방법서」에서 정하는 바에 따라 계약내용 변 경시점의 계약자적립액 및 '
 '미경과보험료를 지급하고, 이 계 약은 더 이상 효력을 가지지 않습니다.'),
    metadata={'source_doc': {'total_pages': 180},
 'doc': {'doc_type': 'terms',
         'file_name': 'meritz_3_petpermint_dog_family_51_86_87_230.pdf',
         'insurer_code': 'meritz',
         'product_code': '3',
         'product_name': '무배당 펫퍼민트 Puppy&Family보험 다이렉트2601',
         'total_pages': 180,
         'page': 85},
 'term_type': 'basic',
 'clause': {'clause_type': 'coverage', 'risk_domains': ['digestive']},
 'indexing': {'chunk_id': 'chunk_000170',
              'chunk_char_len': 181,
              'embedding_model': 'solar-embedding-1-large',
              'tag_method': 'rule',
              'tag_confidence': 0.55}},
)
