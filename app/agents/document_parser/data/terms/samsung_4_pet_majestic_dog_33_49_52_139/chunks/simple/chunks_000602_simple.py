from langchain_core.documents import Document

chunk = Document(
    page_content=('. 다만, 보험가입금액(배상책임의 경우 보상한도액)을 감액할 때 해약환급 금이 없거나 최초 가입할 때 안내한 해약환급금보다 적어질 수 '
 '있습니다.'),
    metadata={'source_doc': {'total_pages': 107},
 'doc': {'doc_type': 'terms',
         'file_name': 'samsung_4_pet_majestic_dog_33_49_52_139.pdf',
         'insurer_code': 'samsung',
         'product_code': '4',
         'product_name': '(장기)무배당 삼성화재 펫보험 위풍댕댕',
         'total_pages': 91,
         'page': 106},
 'term_type': 'special',
 'clause': {'clause_type': 'coverage', 'risk_domains': ['other']},
 'indexing': {'chunk_id': 'chunk_000602',
              'chunk_char_len': 80,
              'embedding_model': 'solar-embedding-1-large',
              'tag_method': 'rule',
              'tag_confidence': 0.8}},
)
