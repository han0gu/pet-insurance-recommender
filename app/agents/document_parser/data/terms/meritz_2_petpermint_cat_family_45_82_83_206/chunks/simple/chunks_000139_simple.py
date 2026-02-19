from langchain_core.documents import Document

chunk = Document(
    page_content=('\uf000 제1항에서 정한 계약의 부활이 이루어진 경우라도 계약 자 또는 피보험자가 최초계약 청약시(2회 이상 부활이 이루 어진 경우 '
 '종전 모든 부활 청약 포함) 제15조(계약 전 알릴 의무)를 위반한 경우에는 제17조(알릴 의무 위반의 효과)가 적용됩니다.\n'
 '제31조(강제집행 등으로 인하여 해지된 계약의 특별부활(효 력회복))'),
    metadata={'source_doc': {'total_pages': 160},
 'doc': {'doc_type': 'terms',
         'file_name': 'meritz_2_petpermint_cat_family_45_82_83_206.pdf',
         'insurer_code': 'meritz',
         'product_code': '2',
         'product_name': '무배당 펫퍼민트 Puppy&Family보험 다이렉트2601',
         'total_pages': 160,
         'page': 74},
 'term_type': 'basic',
 'clause': {'clause_type': 'other', 'risk_domains': ['other']},
 'indexing': {'chunk_id': 'chunk_000139',
              'chunk_char_len': 179,
              'embedding_model': 'solar-embedding-1-large',
              'tag_method': 'llm',
              'tag_confidence': 0.95}},
)
