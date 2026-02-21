from langchain_core.documents import Document

chunk = Document(
    page_content=('| 1 | 뒷다리 근골격계 질환 | NAA024 | 무릎뼈 탈구 |\n'
 '| 1 | 뒷다리 근골격계 질환 | NAA025 NAA026 | 십자 인대 손상 파열 (전방 / 후방) 골절 (뒷다리) |\n'
 '| 2 | 눈 및 부속 기관의 질환 | AIA001 | 눈 및 부속 기관의 양성 신생물 |\n'
 '| 2 | 눈 및 부속 기관의 질환 | AIB001 | 눈 및 부속 기관의 악성 신생물 |\n'
 '| 2 | 눈 및 부속 기관의 질환 | AIC001 | 눈 및 부속 기관의 신생물 (양성 또는 악성이 불확실한) |'),
    metadata={'source_doc': {'total_pages': 180},
 'doc': {'doc_type': 'terms',
         'file_name': 'meritz_3_petpermint_dog_family_51_86_87_230.pdf',
         'insurer_code': 'meritz',
         'product_code': '3',
         'product_name': '무배당 펫퍼민트 Puppy&Family보험 다이렉트2601',
         'total_pages': 1,
         'page': 1},
 'term_type': 'unknown',
 'clause': {'clause_type': 'other', 'risk_domains': ['eye', 'joint']},
 'indexing': {'chunk_id': 'chunk_000558',
              'chunk_char_len': 273,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'normal',
              'tag_method': 'rule',
              'tag_confidence': 0.25}},
)
