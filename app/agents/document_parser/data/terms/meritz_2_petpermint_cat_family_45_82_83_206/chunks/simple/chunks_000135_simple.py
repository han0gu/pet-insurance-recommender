from langchain_core.documents import Document

chunk = Document(
    page_content=('\uf000 제1항에 따라 계약이 해지된 경우에는 제35조(해약환급 금) 제1항에 따른 해약환급금을 계약자에게 지급합니다.\n'
 '제30조(보험료의 납입을 연체하여 해지된 계약의 부활(효력 회복))'),
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
 'indexing': {'chunk_id': 'chunk_000135',
              'chunk_char_len': 101,
              'embedding_model': 'solar-embedding-1-large',
              'tag_method': 'llm',
              'tag_confidence': 0.95}},
)
