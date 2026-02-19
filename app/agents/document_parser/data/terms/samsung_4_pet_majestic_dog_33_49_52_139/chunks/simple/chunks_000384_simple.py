from langchain_core.documents import Document

chunk = Document(
    page_content=('. ④ 제1항에도 불구하고 동일한 상해사고를 직접적인 원인으로 두 종류 이상의 상해 입원 수술을 받거나 같은 종류의 상해 입원 수술을 '
 '2회 이상 받은 경우에는 하나의 상해 입원 수술비만 지급합니다. ⑤ 제2항에도 불구하고 동일한 상해사고를 직접적인 원인으로 두 종류 '
 '이상의 상해 통원'),
    metadata={'source_doc': {'total_pages': 107},
 'doc': {'doc_type': 'terms',
         'file_name': 'samsung_4_pet_majestic_dog_33_49_52_139.pdf',
         'insurer_code': 'samsung',
         'product_code': '4',
         'product_name': '(장기)무배당 삼성화재 펫보험 위풍댕댕',
         'total_pages': 91,
         'page': 75},
 'term_type': 'special',
 'clause': {'clause_type': 'exclusion',
            'risk_domains': ['head', 'joint', 'other']},
 'indexing': {'chunk_id': 'chunk_000384',
              'chunk_char_len': 158,
              'embedding_model': 'solar-embedding-1-large',
              'tag_method': 'llm',
              'tag_confidence': 0.95}},
)
