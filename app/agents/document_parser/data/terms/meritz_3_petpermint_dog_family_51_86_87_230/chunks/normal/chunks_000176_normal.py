from langchain_core.documents import Document

chunk = Document(
    page_content=('제48조(예금보험에 의한 지급보장)\n'
 '회사가 파산 등으로 인하여 보험금 등을 지급하지 못할 경 우에는 예금자보호법에서 정하는 바에 따라 그 지급을 보장 합니다.\n'
 '【예금자보호제도】'),
    metadata={'source_doc': {'total_pages': 180},
 'doc': {'doc_type': 'terms',
         'file_name': 'meritz_3_petpermint_dog_family_51_86_87_230.pdf',
         'insurer_code': 'meritz',
         'product_code': '3',
         'product_name': '무배당 펫퍼민트 Puppy&Family보험 다이렉트2601',
         'total_pages': 180,
         'page': 86},
 'term_type': 'basic',
 'clause': {'clause_type': 'coverage', 'risk_domains': ['digestive']},
 'indexing': {'chunk_id': 'chunk_000176',
              'chunk_char_len': 98,
              'embedding_model': 'solar-embedding-1-large',
              'tag_method': 'rule',
              'tag_confidence': 0.55}},
)
