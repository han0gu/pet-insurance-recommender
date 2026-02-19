from langchain_core.documents import Document

chunk = Document(
    page_content=('. ② 제1항의 상해 골절 수술비는 매사고마다 지급합니다. 다만, 동일한 상해사고를 직접적 인 원인으로 동시에 2가지 이상 또는 2회 '
 '이상의 수술을 받은 경우에는 1회에 한하여 보상합니다.'),
    metadata={'source_doc': {'total_pages': 107},
 'doc': {'doc_type': 'terms',
         'file_name': 'samsung_4_pet_majestic_dog_33_49_52_139.pdf',
         'insurer_code': 'samsung',
         'product_code': '4',
         'product_name': '(장기)무배당 삼성화재 펫보험 위풍댕댕',
         'total_pages': 91,
         'page': 77},
 'term_type': 'special',
 'clause': {'clause_type': 'coverage', 'risk_domains': ['joint', 'head']},
 'indexing': {'chunk_id': 'chunk_000403',
              'chunk_char_len': 105,
              'embedding_model': 'solar-embedding-1-large',
              'tag_method': 'llm',
              'tag_confidence': 0.95}},
)
