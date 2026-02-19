from langchain_core.documents import Document

chunk = Document(
    page_content=('. 1. 1 시행)를 말하며 이후 한국표준질병·사인분류가 개정되는 경우는 개정된 기준에 따라 이 약관에서 보장하는 질병(상병) 해당 '
 '여부를 판단합 니다.'),
    metadata={'source_doc': {'total_pages': 107},
 'doc': {'doc_type': 'terms',
         'file_name': 'samsung_4_pet_majestic_dog_33_49_52_139.pdf',
         'insurer_code': 'samsung',
         'product_code': '4',
         'product_name': '(장기)무배당 삼성화재 펫보험 위풍댕댕',
         'total_pages': 91,
         'page': 33},
 'term_type': 'basic',
 'clause': {'clause_type': 'coverage', 'risk_domains': ['digestive']},
 'indexing': {'chunk_id': 'chunk_000004',
              'chunk_char_len': 85,
              'embedding_model': 'solar-embedding-1-large',
              'tag_method': 'rule',
              'tag_confidence': 0.8}},
)
