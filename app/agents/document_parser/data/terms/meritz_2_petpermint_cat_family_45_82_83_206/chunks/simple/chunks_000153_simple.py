from langchain_core.documents import Document

chunk = Document(
    page_content=('. \uf000 회사는 경과기간별 해약환급금에 관한 표를 계약자에게 제공하여 드립니다. \uf000 제32조의1(위법계약의 해지)에 따라 '
 '위법계약이 해지되 는 경우 회사가 적립한 해지 당시의 계약자적립액 및 미경 과보험료를 반환하여 드립니다.'),
    metadata={'source_doc': {'total_pages': 160},
 'doc': {'doc_type': 'terms',
         'file_name': 'meritz_2_petpermint_cat_family_45_82_83_206.pdf',
         'insurer_code': 'meritz',
         'product_code': '2',
         'product_name': '무배당 펫퍼민트 Puppy&Family보험 다이렉트2601',
         'total_pages': 160,
         'page': 77},
 'term_type': 'basic',
 'clause': {'clause_type': 'other', 'risk_domains': ['other']},
 'indexing': {'chunk_id': 'chunk_000153',
              'chunk_char_len': 125,
              'embedding_model': 'solar-embedding-1-large',
              'tag_method': 'llm',
              'tag_confidence': 0.95}},
)
