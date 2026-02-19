from langchain_core.documents import Document

chunk = Document(
    page_content=('. ② 회사가 청약과 함께 제1회 보험료를 받고 청약을 승낙하기 전에 보험금 지급사유가 발생하였을 때에도 보장개시일부터 이 약관이 정하는 '
 '바에 따라 보장을 합니다.'),
    metadata={'source_doc': {'total_pages': 107},
 'doc': {'doc_type': 'terms',
         'file_name': 'samsung_4_pet_majestic_dog_33_49_52_139.pdf',
         'insurer_code': 'samsung',
         'product_code': '4',
         'product_name': '(장기)무배당 삼성화재 펫보험 위풍댕댕',
         'total_pages': 91,
         'page': 44},
 'term_type': 'basic',
 'clause': {'clause_type': 'coverage', 'risk_domains': ['digestive']},
 'indexing': {'chunk_id': 'chunk_000121',
              'chunk_char_len': 91,
              'embedding_model': 'solar-embedding-1-large',
              'tag_method': 'rule',
              'tag_confidence': 0.8}},
)
