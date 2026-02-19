from langchain_core.documents import Document

chunk = Document(
    page_content=('. \uf000 회사가 청약과 함께 제1회 보험료를 받고 청약을 승낙하 기 전에 보험금 지급사유가 발생하였을 때에도 보장개시일 부터 이 '
 '약관이 정하는 바에 따라 보장을 합니다.'),
    metadata={'source_doc': {'total_pages': 180},
 'doc': {'doc_type': 'terms',
         'file_name': 'meritz_3_petpermint_dog_family_51_86_87_230.pdf',
         'insurer_code': 'meritz',
         'product_code': '3',
         'product_name': '무배당 펫퍼민트 Puppy&Family보험 다이렉트2601',
         'total_pages': 180,
         'page': 75},
 'term_type': 'basic',
 'clause': {'clause_type': 'coverage', 'risk_domains': ['digestive']},
 'indexing': {'chunk_id': 'chunk_000119',
              'chunk_char_len': 93,
              'embedding_model': 'solar-embedding-1-large',
              'tag_method': 'rule',
              'tag_confidence': 0.55}},
)
