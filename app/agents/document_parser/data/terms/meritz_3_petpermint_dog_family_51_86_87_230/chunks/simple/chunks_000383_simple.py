from langchain_core.documents import Document

chunk = Document(
    page_content=('보험기간이 만료된 경우에도 만료일부터 180일 이내의 치료 비는 제2항에 따라 보상하여 드립니다. 다만, 사고일 또는 발병일부터 365일 '
 '이내인 경우에 한합니다. \uf000「반려동물 비용손해 관련 특별약관 일반조항」제15조(재 가입) 제6항에 따라 보험계약이 연장된 '
 '경우에는 종전 계약 의 보험기간을 연장하는 것으로 보아 제6항을 적용하지 않 습니다. \uf000 부활(효력회복)되는 이 계약의 '
 '보장개시는「반려동물 비 용손해 관련 특별약관 일반조항」제18조(보험료의 납입을 연체하여 해지된 계약의 부활(효력회복))를 따릅니다'),
    metadata={'source_doc': {'total_pages': 180},
 'doc': {'doc_type': 'terms',
         'file_name': 'meritz_3_petpermint_dog_family_51_86_87_230.pdf',
         'insurer_code': 'meritz',
         'product_code': '3',
         'product_name': '무배당 펫퍼민트 Puppy&Family보험 다이렉트2601',
         'total_pages': 180,
         'page': 130},
 'term_type': 'special',
 'clause': {'clause_type': 'coverage', 'risk_domains': ['digestive']},
 'indexing': {'chunk_id': 'chunk_000383',
              'chunk_char_len': 284,
              'embedding_model': 'solar-embedding-1-large',
              'tag_method': 'rule',
              'tag_confidence': 0.8}},
)
