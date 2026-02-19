from langchain_core.documents import Document

chunk = Document(
    page_content=('① 재가입일에 있어서 반려동물의 나이가 회사가 최초가 입 당시 정한 재가입 나이의 범위 내일 것 ② 재가입 전 계약의 보험료가 정상적으로 '
 '납입완료 되었 을 것'),
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
 'indexing': {'chunk_id': 'chunk_000249',
              'chunk_char_len': 88,
              'embedding_model': 'solar-embedding-1-large',
              'tag_method': 'rule',
              'tag_confidence': 0.55}},
)
