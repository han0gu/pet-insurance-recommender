from langchain_core.documents import Document

chunk = Document(
    page_content=('. 회사가 부활(효력회복)을 승낙한 때에 계약자는 부활 (효력회복)을 청약한 날까지의 연체된 보험료와 이에 대한 연체된 이자(보장보험료에 '
 '대해서 평균공시이율+1%로 계산 한 이자)를 더하여 납입하여야 합니다.'),
    metadata={'source_doc': {'total_pages': 160},
 'doc': {'doc_type': 'terms',
         'file_name': 'meritz_2_petpermint_cat_family_45_82_83_206.pdf',
         'insurer_code': 'meritz',
         'product_code': '2',
         'product_name': '무배당 펫퍼민트 Puppy&Family보험 다이렉트2601',
         'total_pages': 160,
         'page': 74},
 'term_type': 'basic',
 'clause': {'clause_type': 'coverage', 'risk_domains': ['digestive']},
 'indexing': {'chunk_id': 'chunk_000137',
              'chunk_char_len': 116,
              'embedding_model': 'solar-embedding-1-large',
              'tag_method': 'rule',
              'tag_confidence': 0.8}},
)
