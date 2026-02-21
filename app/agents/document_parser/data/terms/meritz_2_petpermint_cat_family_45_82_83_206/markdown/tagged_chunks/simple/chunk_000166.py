from langchain_core.documents import Document

chunk = Document(
    page_content=('하는 다른 계약(공제계약을 포함합니다)이 있을 경우 각 계\n'
 '약에 대하여 다른 계약이 없는 것으로 하여 각각 산출한 지\n'
 '급보험금의 합계액이 피보험자가 부담한 비용금액을 초과할\n'
 '때에는 아래에 따라 보험금을 지급합니다.피보험자가 이 계약의 지급보험금\n'
 '부담한 총 × 다른 계약이 없는 것으로 하여 각각 계산한\n'
 '비용금액 지급보험금의 합계액\uf000 피보험자가 다른 계약에 대하여 보험금 청구를 포기한\n'
 '경우에도 회사의 제1항에 따른 지급보험금 결정에는 영향을'),
    metadata={'source_doc': {'total_pages': 160},
 'doc': {'doc_type': 'terms',
         'file_name': 'meritz_2_petpermint_cat_family_45_82_83_206.pdf',
         'insurer_code': 'meritz',
         'product_code': '2',
         'product_name': '무배당 펫퍼민트 Cat&Family보험 다이렉트2601',
         'total_pages': 1,
         'page': 1},
 'term_type': 'unknown',
 'clause': {'clause_type': 'other', 'risk_domains': ['other']},
 'indexing': {'chunk_id': 'chunk_000166',
              'chunk_char_len': 246,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'simple',
              'tag_method': 'rule',
              'tag_confidence': 0.25}},
)
