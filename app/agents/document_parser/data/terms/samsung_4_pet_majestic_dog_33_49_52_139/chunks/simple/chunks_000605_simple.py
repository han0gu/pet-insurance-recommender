from langchain_core.documents import Document

chunk = Document(
    page_content=('. 다만, 반려동물의 나이 및 품 종이 정정되기 이전에는 「나이 및 품종이 정정되기 전에 적용된 보험료율」의 「나 이 및 품종이 정정된 '
 '후에 적용해야할 보험료율」에 대한 비율에 따라 보험금을 삭감 하여 지급합니다.'),
    metadata={'source_doc': {'total_pages': 107},
 'doc': {'doc_type': 'terms',
         'file_name': 'samsung_4_pet_majestic_dog_33_49_52_139.pdf',
         'insurer_code': 'samsung',
         'product_code': '4',
         'product_name': '(장기)무배당 삼성화재 펫보험 위풍댕댕',
         'total_pages': 91,
         'page': 106},
 'term_type': 'special',
 'clause': {'clause_type': 'exclusion', 'risk_domains': ['other']},
 'indexing': {'chunk_id': 'chunk_000605',
              'chunk_char_len': 119,
              'embedding_model': 'solar-embedding-1-large',
              'tag_method': 'llm',
              'tag_confidence': 0.95}},
)
