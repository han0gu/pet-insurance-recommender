from langchain_core.documents import Document

chunk = Document(
    page_content=('\uf000 제8항 내지 제10항에 따라 계약이 해지된 경우 회사는 \uf000 보통약관 제35조(해약환급금) 제1항에 따른 해약환급금을 '
 '계약자에게 지급합니다.\n'
 '제16조(제1회 보험료 및 회사의 보장개시)'),
    metadata={'source_doc': {'total_pages': 160},
 'doc': {'doc_type': 'terms',
         'file_name': 'meritz_2_petpermint_cat_family_45_82_83_206.pdf',
         'insurer_code': 'meritz',
         'product_code': '2',
         'product_name': '무배당 펫퍼민트 Puppy&Family보험 다이렉트2601',
         'total_pages': 160,
         'page': 99},
 'term_type': 'special',
 'clause': {'clause_type': 'coverage', 'risk_domains': ['digestive']},
 'indexing': {'chunk_id': 'chunk_000259',
              'chunk_char_len': 104,
              'embedding_model': 'solar-embedding-1-large',
              'tag_method': 'rule',
              'tag_confidence': 0.8}},
)
