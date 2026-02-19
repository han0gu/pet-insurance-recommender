from langchain_core.documents import Document

chunk = Document(
    page_content=('\uf000 제8항에 따라 계약자에게 해지된다는 사실을 알려드린 최초시점부터 90일 이내에 계약자의 재가입 의사가 확인되 지 않는 경우 '
 '해당 시점부터 계약은 해지됩니다.'),
    metadata={'source_doc': {'total_pages': 180},
 'doc': {'doc_type': 'terms',
         'file_name': 'meritz_3_petpermint_dog_family_51_86_87_230.pdf',
         'insurer_code': 'meritz',
         'product_code': '3',
         'product_name': '무배당 펫퍼민트 Puppy&Family보험 다이렉트2601',
         'total_pages': 180,
         'page': 103},
 'term_type': 'special',
 'clause': {'clause_type': 'other', 'risk_domains': ['other']},
 'indexing': {'chunk_id': 'chunk_000252',
              'chunk_char_len': 89,
              'embedding_model': 'solar-embedding-1-large',
              'tag_method': 'llm',
              'tag_confidence': 0.95}},
)
