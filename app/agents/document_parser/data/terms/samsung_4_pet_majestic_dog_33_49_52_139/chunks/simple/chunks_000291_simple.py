from langchain_core.documents import Document

chunk = Document(
    page_content=('⑦ 제1항에 따라 계약이 해지된 경우에는 제35조(해약환급금)에서 정한 해약환급금을 계 약자에게 지급합니다.\n'
 '제 30조 (보험료의 납입을 연체하여 해지된 특별약관의 부활(효력회복))'),
    metadata={'source_doc': {'total_pages': 107},
 'doc': {'doc_type': 'terms',
         'file_name': 'samsung_4_pet_majestic_dog_33_49_52_139.pdf',
         'insurer_code': 'samsung',
         'product_code': '4',
         'product_name': '(장기)무배당 삼성화재 펫보험 위풍댕댕',
         'total_pages': 91,
         'page': 62},
 'term_type': 'special',
 'clause': {'clause_type': 'other', 'risk_domains': ['other']},
 'indexing': {'chunk_id': 'chunk_000291',
              'chunk_char_len': 101,
              'embedding_model': 'solar-embedding-1-large',
              'tag_method': 'llm',
              'tag_confidence': 0.95}},
)
