from langchain_core.documents import Document

chunk = Document(
    page_content=('. ③ 회사는 피보험자가 이 특별약관에 적합하지 않은 경우에는 승낙을 거절하거나 별도의 조건(보험가입금액 제한, 일부보장 제외, 보험금 '
 '삭감, 보험료 할증 등)을 붙여 승낙할 수 있습니다.'),
    metadata={'source_doc': {'total_pages': 107},
 'doc': {'doc_type': 'terms',
         'file_name': 'samsung_4_pet_majestic_dog_33_49_52_139.pdf',
         'insurer_code': 'samsung',
         'product_code': '4',
         'product_name': '(장기)무배당 삼성화재 펫보험 위풍댕댕',
         'total_pages': 91,
         'page': 58},
 'term_type': 'special',
 'clause': {'clause_type': 'exclusion', 'risk_domains': ['digestive']},
 'indexing': {'chunk_id': 'chunk_000240',
              'chunk_char_len': 105,
              'embedding_model': 'solar-embedding-1-large',
              'tag_method': 'rule',
              'tag_confidence': 0.85}},
)
