from langchain_core.documents import Document

chunk = Document(
    page_content=('일부에 대하여 나누어 지급받거나 일시에 지급받는 방법으 로 변경할 수 있습니다. \uf000 회사는 제1항에 따라 일시에 지급할 금액을 '
 '나누어 지급 하는 경우에는 나중에 지급할 금액에 대하여 평균공시이율 을 연단위 복리로 계산한 금액을 더하며, 나누어 지급할 금 액을 '
 '일시에 지급하는 경우에는 평균공시이율을 연단위 복 리로 할인한 금액을 지급합니다.\n'
 '【보험금 지급 예시】'),
    metadata={'source_doc': {'total_pages': 160},
 'doc': {'doc_type': 'terms',
         'file_name': 'meritz_2_petpermint_cat_family_45_82_83_206.pdf',
         'insurer_code': 'meritz',
         'product_code': '2',
         'product_name': '무배당 펫퍼민트 Puppy&Family보험 다이렉트2601',
         'total_pages': 160,
         'page': 56},
 'term_type': 'basic',
 'clause': {'clause_type': 'coverage', 'risk_domains': ['digestive']},
 'indexing': {'chunk_id': 'chunk_000043',
              'chunk_char_len': 202,
              'embedding_model': 'solar-embedding-1-large',
              'tag_method': 'rule',
              'tag_confidence': 0.8}},
)
