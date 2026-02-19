from langchain_core.documents import Document

chunk = Document(
    page_content=('① 이 특별약관에서 「수술」 이라 함은 의사에 의하여 치료가 필요하다고 인정된 경우로 서 자택 등에서 치료가 곤란하여 의료법 '
 '제3조(의료기관)에서 규정한 병원, 의원 또는 국외의 의료관련법에서 정한 의료기관에서 의사의 관리 하에 치료를 직접적인 목적으 로 기구를 '
 '사용하여 생체(生體)에 절단(切断, 특정부위를 잘라 내는 것), 절제(切除, 특 정부위를 잘라 없애는 것) 등의 조작을 가하는 것을 '
 '말합니다'),
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
 'indexing': {'chunk_id': 'chunk_000390',
              'chunk_char_len': 225,
              'embedding_model': 'solar-embedding-1-large',
              'tag_method': 'llm',
              'tag_confidence': 0.95}},
)
