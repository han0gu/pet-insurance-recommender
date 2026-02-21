from langchain_core.documents import Document

chunk = Document(
    page_content=('또는 섭취한 결과로 생긴 중독 증상을 포함합니다. 그러나 음식물 섭취로 인한 증\n'
 '상, 세균성 음식물 중독과 상습적으로 흡입, 흡수 또는 섭취한 결과로 생긴 중독\n'
 '증상은 포함되지 않습니다.# <용어풀이># [음식물]반려견이 일상 생활 중 보호자 또는 생산자의 의도와 상관 없이 섭취할 수 있는 모든 '
 '식이 원료와\n'
 '가공품 및 부산물(뼈, 과일 씨 등 폐기 대상물질)을 말하며, 사람 및 다른 동물의 식이로 활용될 수'),
    metadata={'source_doc': {'total_pages': 107},
 'doc': {'doc_type': 'terms',
         'file_name': 'samsung_4_pet_majestic_dog_33_49_52_139.pdf',
         'insurer_code': 'samsung',
         'product_code': '4',
         'product_name': '(장기)무배당 삼성화재 펫보험 위풍댕댕',
         'total_pages': 1,
         'page': 1},
 'term_type': 'unknown',
 'clause': {'clause_type': 'other', 'risk_domains': ['other']},
 'indexing': {'chunk_id': 'chunk_000448',
              'chunk_char_len': 229,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'normal',
              'tag_method': 'rule',
              'tag_confidence': 0.25}},
)
