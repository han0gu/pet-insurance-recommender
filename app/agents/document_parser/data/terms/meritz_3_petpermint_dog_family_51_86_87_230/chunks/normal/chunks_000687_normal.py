from langchain_core.documents import Document

chunk = Document(
    page_content=('QCA001 | 귀 가려움증 (원인 불명)\n'
 'QFA001 | 발진 (원인 불명)\n'
 'QFA002 | 피부염 (원인 불명)\n'
 'QFA003 | 피부의 가려움증 (원인 불명)\n'
 'QFA004 | 탈모 (원인 불명)'),
    metadata={'source_doc': {'total_pages': 180},
 'doc': {'doc_type': 'terms',
         'file_name': 'meritz_3_petpermint_dog_family_51_86_87_230.pdf',
         'insurer_code': 'meritz',
         'product_code': '3',
         'product_name': '무배당 펫퍼민트 Puppy&Family보험 다이렉트2601',
         'total_pages': 180,
         'page': 198},
 'term_type': 'special',
 'clause': {'clause_type': 'definition', 'risk_domains': ['skin']},
 'indexing': {'chunk_id': 'chunk_000687',
              'chunk_char_len': 110,
              'embedding_model': 'solar-embedding-1-large',
              'tag_method': 'llm',
              'tag_confidence': 0.95}},
)
