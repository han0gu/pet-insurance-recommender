from langchain_core.documents import Document

chunk = Document(
    page_content=('. 전자적 상품설명장치에 안내의 속도와 음량을 조절할 수 있는 기능을 갖출 것 5. 제3호 및 제4호의 내용에 관한 사항을 계약자에게 '
 '안내할 것'),
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
 'indexing': {'chunk_id': 'chunk_000290',
              'chunk_char_len': 80,
              'embedding_model': 'solar-embedding-1-large',
              'tag_method': 'llm',
              'tag_confidence': 0.95}},
)
