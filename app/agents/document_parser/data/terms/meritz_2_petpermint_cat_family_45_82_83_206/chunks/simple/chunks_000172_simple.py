from langchain_core.documents import Document

chunk = Document(
    page_content=('가입금액 및 납입보험료가 변경될 수 있으며, 계약내용 변 경 시점 이후 잔여보험기간의 보장을 위한 재원인 계약자적 립액 및 미경과보험료 '
 '정산으로 계약자가 추가로 납입하여 야 할(또는 반환받을) 금액이 발생할 수 있습니다.'),
    metadata={'source_doc': {'total_pages': 160},
 'doc': {'doc_type': 'terms',
         'file_name': 'meritz_2_petpermint_cat_family_45_82_83_206.pdf',
         'insurer_code': 'meritz',
         'product_code': '2',
         'product_name': '무배당 펫퍼민트 Puppy&Family보험 다이렉트2601',
         'total_pages': 160,
         'page': 81},
 'term_type': 'basic',
 'clause': {'clause_type': 'coverage', 'risk_domains': ['digestive']},
 'indexing': {'chunk_id': 'chunk_000172',
              'chunk_char_len': 123,
              'embedding_model': 'solar-embedding-1-large',
              'tag_method': 'rule',
              'tag_confidence': 0.8}},
)
