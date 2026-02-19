from langchain_core.documents import Document

chunk = Document(
    page_content=('제38조 (배당금의 지급)\n'
 '회사는 이 보험에 대하여 계약자에게 배당금을 지급하지 않습니다.\n'
 '제7관 분쟁의 조정 등\n'
 '제 39조 (분쟁의 조정)'),
    metadata={'source_doc': {'total_pages': 107},
 'doc': {'doc_type': 'terms',
         'file_name': 'samsung_4_pet_majestic_dog_33_49_52_139.pdf',
         'insurer_code': 'samsung',
         'product_code': '4',
         'product_name': '(장기)무배당 삼성화재 펫보험 위풍댕댕',
         'total_pages': 91,
         'page': 48},
 'term_type': 'basic',
 'clause': {'clause_type': 'exclusion', 'risk_domains': ['other']},
 'indexing': {'chunk_id': 'chunk_000160',
              'chunk_char_len': 78,
              'embedding_model': 'solar-embedding-1-large',
              'tag_method': 'rule',
              'tag_confidence': 0.85}},
)
