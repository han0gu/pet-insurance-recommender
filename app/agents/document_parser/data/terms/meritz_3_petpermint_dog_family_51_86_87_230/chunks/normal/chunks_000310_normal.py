from langchain_core.documents import Document

chunk = Document(
    page_content=('. \uf000 부활(효력회복)되는 이 계약의 보장개시는「반려동물 비 용손해 관련 특별약관 일반조항」제18조(보험료의 납입을 연체하여 '
 '해지된 계약의 부활(효력회복))를 따릅니다. 이 경우 부활(효력회복)일을 계약일로 하여 제3항 및 제4항의 보장개시일을 적용합니다.'),
    metadata={'source_doc': {'total_pages': 180},
 'doc': {'doc_type': 'terms',
         'file_name': 'meritz_3_petpermint_dog_family_51_86_87_230.pdf',
         'insurer_code': 'meritz',
         'product_code': '3',
         'product_name': '무배당 펫퍼민트 Puppy&Family보험 다이렉트2601',
         'total_pages': 180,
         'page': 114},
 'term_type': 'special',
 'clause': {'clause_type': 'coverage', 'risk_domains': ['digestive']},
 'indexing': {'chunk_id': 'chunk_000310',
              'chunk_char_len': 143,
              'embedding_model': 'solar-embedding-1-large',
              'tag_method': 'rule',
              'tag_confidence': 0.55}},
)
