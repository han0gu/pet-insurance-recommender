from langchain_core.documents import Document

chunk = Document(
    page_content=('【 예시 】\n'
 '입원특약에 가입한 피보험자가 20일간 입원하였음에도 불 구하고 입원확인서를 변조하여 입원일수 30일에 해당하는 보험금을 청구한 경우, '
 '회사는 그 사실을 안 날로부터 1 개월 이내에 계약을 해지할 수 있습니다. 다만, 이 경우 에도 회사는 입원일수 20일에 해당하는 '
 '보험금을 지급합 니다.\n'
 '\uf000 회사가 제1항에 따라 계약을 해지한 경우 회사는 그 취 지를 계약자에게 통지하고 제35조(해약환급금) 제1항에 따 른 '
 '해약환급금을 지급합니다.\n'
 '제34조(회사의 파산선고와 해지)'),
    metadata={'source_doc': {'total_pages': 160},
 'doc': {'doc_type': 'terms',
         'file_name': 'meritz_2_petpermint_cat_family_45_82_83_206.pdf',
         'insurer_code': 'meritz',
         'product_code': '2',
         'product_name': '무배당 펫퍼민트 Puppy&Family보험 다이렉트2601',
         'total_pages': 160,
         'page': 77},
 'term_type': 'basic',
 'clause': {'clause_type': 'exclusion', 'risk_domains': ['other']},
 'indexing': {'chunk_id': 'chunk_000150',
              'chunk_char_len': 268,
              'embedding_model': 'solar-embedding-1-large',
              'tag_method': 'llm',
              'tag_confidence': 0.9}},
)
