from langchain_core.documents import Document

chunk = Document(
    page_content=('- 조(계약 전 알릴 의무)를 위반하고 그 의무가 중요한\n'
 '- 사항에 해당하는 경우\n'
 '- ② 뚜렷한 위험의 증가와 관련된 제8조(계약 후 알릴 의\n'
 '- 무) 제1항에서 정한 계약 후 알릴 의무를 계약자 또는\n'
 '- 피보험자의 고의 또는 중대한 과실로 이행하지 않았을\n'
 '- 때\n'
 '\uf000 제1항 제1호의 경우에도 불구하고 다음 중 하나에 해당\n'
 '하는 경우에는 회사는 계약을 해지할 수 없습니다.- ① 회사가 최초계약 체결당시에 그 사실을 알았거나 과실\n'
 '- 로 인하여 알지 못하였을 때\n'
 '- ② 회사가 그 사실을 안 날부터 1개월 이상 지났거나 또'),
    metadata={'source_doc': {'total_pages': 160},
 'doc': {'doc_type': 'terms',
         'file_name': 'meritz_2_petpermint_cat_family_45_82_83_206.pdf',
         'insurer_code': 'meritz',
         'product_code': '2',
         'product_name': '무배당 펫퍼민트 Cat&Family보험 다이렉트2601',
         'total_pages': 1,
         'page': 1},
 'term_type': 'unknown',
 'clause': {'clause_type': 'other', 'risk_domains': ['digestive']},
 'indexing': {'chunk_id': 'chunk_000176',
              'chunk_char_len': 292,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'simple',
              'tag_method': 'rule',
              'tag_confidence': 0.25}},
)
