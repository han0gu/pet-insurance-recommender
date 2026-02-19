from langchain_core.documents import Document

chunk = Document(
    page_content=('\uf000 제2항에 따라 추가적인 조사가 이루어지는 경우, 회사는 피보험자의 청구에 따라 회사가 추정하는 보험금의 50% 상 당액을 '
 '가지급보험금으로 지급합니다.\n'
 '【가지급보험금】\n'
 '보험금이 지급기한 내에 지급되지 못할 것으로 판단되는 경우 회사가 예상되는 보험금의 일부를 먼저 지급하는 제도로 피보험자가 필요로 하는 '
 '비용을 보전해 주기 위 해 회사가 먼저 지급하는 임시 교부금을 말합니다.\n'
 '\uf000 회사는 제1항에서 정한 지급기일내에 보험금을 지급하지'),
    metadata={'source_doc': {'total_pages': 160},
 'doc': {'doc_type': 'terms',
         'file_name': 'meritz_2_petpermint_cat_family_45_82_83_206.pdf',
         'insurer_code': 'meritz',
         'product_code': '2',
         'product_name': '무배당 펫퍼민트 Puppy&Family보험 다이렉트2601',
         'total_pages': 160,
         'page': 89},
 'term_type': 'special',
 'clause': {'clause_type': 'claim', 'risk_domains': ['other']},
 'indexing': {'chunk_id': 'chunk_000205',
              'chunk_char_len': 243,
              'embedding_model': 'solar-embedding-1-large',
              'tag_method': 'llm',
              'tag_confidence': 0.95}},
)
