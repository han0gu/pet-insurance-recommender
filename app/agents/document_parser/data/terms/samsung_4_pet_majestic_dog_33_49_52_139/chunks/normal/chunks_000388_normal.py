from langchain_core.documents import Document

chunk = Document(
    page_content=('. ② 이 특별약관에서 「외래」 라 함은 병원 또는 의원의 의사에 의하여 상해의 치료가 필 요하다고 인정된 경우로서, 의료법 '
 '제3조(의료기관)에서 규정한 국내의 병원, 의원 또 는 국외의 의료관련법에서 정한 의료기관에 입실하지 않고 의사의 관리 하에 치료에 '
 '전념하는 것을 말합니다.'),
    metadata={'source_doc': {'total_pages': 107},
 'doc': {'doc_type': 'terms',
         'file_name': 'samsung_4_pet_majestic_dog_33_49_52_139.pdf',
         'insurer_code': 'samsung',
         'product_code': '4',
         'product_name': '(장기)무배당 삼성화재 펫보험 위풍댕댕',
         'total_pages': 91,
         'page': 75},
 'term_type': 'special',
 'clause': {'clause_type': 'definition', 'risk_domains': ['other']},
 'indexing': {'chunk_id': 'chunk_000388',
              'chunk_char_len': 157,
              'embedding_model': 'solar-embedding-1-large',
              'tag_method': 'llm',
              'tag_confidence': 0.95}},
)
