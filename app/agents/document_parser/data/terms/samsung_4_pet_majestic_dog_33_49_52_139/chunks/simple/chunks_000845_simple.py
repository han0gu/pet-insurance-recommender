from langchain_core.documents import Document

chunk = Document(
    page_content=('② 제1항에 따라 보장이 제한되는 범위는 의학적으로 인과관계가 있다고 입증된 경우 또 는 경험통계적으로 인과관계가 유의성있게 입증된 경우 '
 '등 피보험자의 과거 및 현재 병력(계약 전 알릴 의무 사항에 해당하는 질병)과 직접적으로 관련이 있는 신체부위 또는 질병 등으로 '
 '제한하며, 이 특별약관을 부가할 때에는 회사는 부담보 설정범위 및 사유를 계약자에게 설명하여 드립니다'),
    metadata={'source_doc': {'total_pages': 107},
 'doc': {'doc_type': 'terms',
         'file_name': 'samsung_4_pet_majestic_dog_33_49_52_139.pdf',
         'insurer_code': 'samsung',
         'product_code': '4',
         'product_name': '(장기)무배당 삼성화재 펫보험 위풍댕댕',
         'total_pages': 91,
         'page': 136},
 'term_type': 'special',
 'clause': {'clause_type': 'coverage', 'risk_domains': ['digestive']},
 'indexing': {'chunk_id': 'chunk_000845',
              'chunk_char_len': 205,
              'embedding_model': 'solar-embedding-1-large',
              'tag_method': 'rule',
              'tag_confidence': 0.8}},
)
