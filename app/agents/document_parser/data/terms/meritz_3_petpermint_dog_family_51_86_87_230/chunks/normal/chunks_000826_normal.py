from langchain_core.documents import Document

chunk = Document(
    page_content=('. 바) “중증발작”이라 함은 전신경련을 동반하는 발 작으로써 신체의 균형을 유지하지 못하고 쓰러지 는 발작 또는 의식장해가 3분이상 '
 '지속되는 발작 을 말한다. 사) “경증발작”이라 함은 운동장해가 발생하나 스스 로 신체의 균형을 유지할 수 있는 발작 또는 3분 이내에 '
 '정상으로 회복되는 발작을 말한다.'),
    metadata={'source_doc': {'total_pages': 180},
 'doc': {'doc_type': 'terms',
         'file_name': 'meritz_3_petpermint_dog_family_51_86_87_230.pdf',
         'insurer_code': 'meritz',
         'product_code': '3',
         'product_name': '무배당 펫퍼민트 Puppy&Family보험 다이렉트2601',
         'total_pages': 180,
         'page': 229},
 'term_type': 'special',
 'clause': {'clause_type': 'definition', 'risk_domains': ['head']},
 'indexing': {'chunk_id': 'chunk_000826',
              'chunk_char_len': 169,
              'embedding_model': 'solar-embedding-1-large',
              'tag_method': 'llm',
              'tag_confidence': 0.95}},
)
