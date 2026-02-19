from langchain_core.documents import Document

chunk = Document(
    page_content=('. 다만, 재가입 계약이 직전 계약보다 보장내용 및 범위 등이 확대된 경우 확대된 내용 에 대해 회사는 재가입 시점의 인수기준에 따라 '
 '승낙하거나 일부 보장을 제한할 수 있습니다.'),
    metadata={'source_doc': {'total_pages': 160},
 'doc': {'doc_type': 'terms',
         'file_name': 'meritz_2_petpermint_cat_family_45_82_83_206.pdf',
         'insurer_code': 'meritz',
         'product_code': '2',
         'product_name': '무배당 펫퍼민트 Puppy&Family보험 다이렉트2601',
         'total_pages': 160,
         'page': 98},
 'term_type': 'special',
 'clause': {'clause_type': 'renewal', 'risk_domains': ['digestive']},
 'indexing': {'chunk_id': 'chunk_000251',
              'chunk_char_len': 99,
              'embedding_model': 'solar-embedding-1-large',
              'tag_method': 'rule',
              'tag_confidence': 0.55}},
)
