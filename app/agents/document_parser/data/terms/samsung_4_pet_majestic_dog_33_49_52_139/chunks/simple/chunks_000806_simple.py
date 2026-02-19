from langchain_core.documents import Document

chunk = Document(
    page_content=('제도성 특별약관\n'
 '※ 약관에서 인용된 법·규정은 「별표 및 참고」 의 「약관에서 인용된 법·규정」 에서 확인할 수 있습니다.\n'
 '제도성 특별 약관'),
    metadata={'source_doc': {'total_pages': 107},
 'doc': {'doc_type': 'terms',
         'file_name': 'samsung_4_pet_majestic_dog_33_49_52_139.pdf',
         'insurer_code': 'samsung',
         'product_code': '4',
         'product_name': '(장기)무배당 삼성화재 펫보험 위풍댕댕',
         'total_pages': 91,
         'page': 129},
 'term_type': 'special',
 'clause': {'clause_type': 'other', 'risk_domains': ['other']},
 'indexing': {'chunk_id': 'chunk_000806',
              'chunk_char_len': 78,
              'embedding_model': 'solar-embedding-1-large',
              'tag_method': 'llm',
              'tag_confidence': 0.95}},
)
