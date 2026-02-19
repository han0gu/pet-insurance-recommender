from langchain_core.documents import Document

chunk = Document(
    page_content=('. \uf000 부활(효력회복)되는 이 특별약관의 보장개시는「반려동 \uf000 물 비용손해 관련 특별약관 일반조항」제18조(보험료의 납 '
 '입을 연체하여 해지된 계약의 부활(효력회복))를 따릅니다. 이 경우 부활(효력회복)일을 계약일로 하여 제5항 및 제6항 의 보장개시일을 '
 '적용합니다.'),
    metadata={'source_doc': {'total_pages': 160},
 'doc': {'doc_type': 'terms',
         'file_name': 'meritz_2_petpermint_cat_family_45_82_83_206.pdf',
         'insurer_code': 'meritz',
         'product_code': '2',
         'product_name': '무배당 펫퍼민트 Puppy&Family보험 다이렉트2601',
         'total_pages': 160,
         'page': 149},
 'term_type': 'special',
 'clause': {'clause_type': 'coverage', 'risk_domains': ['digestive']},
 'indexing': {'chunk_id': 'chunk_000496',
              'chunk_char_len': 149,
              'embedding_model': 'solar-embedding-1-large',
              'tag_method': 'rule',
              'tag_confidence': 0.8}},
)
