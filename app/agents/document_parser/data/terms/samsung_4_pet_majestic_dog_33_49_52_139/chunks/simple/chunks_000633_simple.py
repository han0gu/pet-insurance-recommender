from langchain_core.documents import Document

chunk = Document(
    page_content=('1. 재가입일에 있어서 반려동물의 나이가 회사가 최초가입 당시 정한 재가입 나이의 범위 내일 것 2. 재가입 전 계약의 보험료가 '
 '정상적으로 납입완료 되었을 것'),
    metadata={'source_doc': {'total_pages': 107},
 'doc': {'doc_type': 'terms',
         'file_name': 'samsung_4_pet_majestic_dog_33_49_52_139.pdf',
         'insurer_code': 'samsung',
         'product_code': '4',
         'product_name': '(장기)무배당 삼성화재 펫보험 위풍댕댕',
         'total_pages': 91,
         'page': 108},
 'term_type': 'special',
 'clause': {'clause_type': 'other', 'risk_domains': ['other']},
 'indexing': {'chunk_id': 'chunk_000633',
              'chunk_char_len': 88,
              'embedding_model': 'solar-embedding-1-large',
              'tag_method': 'llm',
              'tag_confidence': 0.95}},
)
