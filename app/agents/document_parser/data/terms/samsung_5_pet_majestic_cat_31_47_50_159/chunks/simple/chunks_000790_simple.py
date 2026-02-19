from langchain_core.documents import Document

chunk = Document(
    page_content=('제 5조 (갱신일 이후 부활(효력회복)을 청약하는 경우 연체된 보험료의 적용)\n'
 '보통약관 제31조(보험료의 납입을 연체하여 해지된 계약의 부활(효력회복)) 1항에서 정한 연체된 보험료는 갱신일부터 부활(효력회복)을 '
 '청약한 날까지의 납입이 연체된 보험료를 말합니다.\n'
 '제 6조 (갱신계약의 보장내용 변경시 계약자 안내에 관한 사항)'),
    metadata={'source_doc': {'total_pages': 129},
 'doc': {'doc_type': 'terms',
         'file_name': 'samsung_5_pet_majestic_cat_31_47_50_159.pdf',
         'insurer_code': 'samsung',
         'product_code': '5',
         'product_name': '(장기)무배당 삼성화재 펫보험 의기냥냥',
         'total_pages': 107,
         'page': 123},
 'term_type': 'special',
 'clause': {'clause_type': 'coverage', 'risk_domains': ['digestive']},
 'indexing': {'chunk_id': 'chunk_000790',
              'chunk_char_len': 182,
              'embedding_model': 'solar-embedding-1-large',
              'tag_method': 'rule',
              'tag_confidence': 0.8}},
)
