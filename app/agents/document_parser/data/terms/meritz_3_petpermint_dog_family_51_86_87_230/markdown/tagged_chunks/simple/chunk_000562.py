from langchain_core.documents import Document

chunk = Document(
    page_content=('| FEA004 | 백내장 |  |  |\n'
 '| FFA001 | 망막 변성 / 망막 위축 / PRA |  |  |\n'
 '| FFA002 | 망막 박리 (유리체 변성 포함) |  |  |\n'
 '| FGA001 | 녹내장 (좌안) |  |  |\n'
 '| FGA002 | 녹내장 (우안) |  |  |\n'
 '| FGA003 | 동양안충증 |  |  |\n'
 '| FGA004 | 기타 안과 질환 |  |  |\n'
 '| FGA005 | 초자체변성 |  |  |\n'
 '| FGA006 | 상공막염 |  |  |\n'
 '| FGA007 | 녹내장 |  |  |'),
    metadata={'source_doc': {'total_pages': 180},
 'doc': {'doc_type': 'terms',
         'file_name': 'meritz_3_petpermint_dog_family_51_86_87_230.pdf',
         'insurer_code': 'meritz',
         'product_code': '3',
         'product_name': '무배당 펫퍼민트 Puppy&Family보험 다이렉트2601',
         'total_pages': 1,
         'page': 1},
 'term_type': 'unknown',
 'clause': {'clause_type': 'other', 'risk_domains': ['digestive', 'eye']},
 'indexing': {'chunk_id': 'chunk_000562',
              'chunk_char_len': 279,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'simple',
              'tag_method': 'rule',
              'tag_confidence': 0.25}},
)
