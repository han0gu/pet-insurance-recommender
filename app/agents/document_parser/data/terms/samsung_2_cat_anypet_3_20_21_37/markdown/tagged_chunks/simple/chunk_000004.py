from langchain_core.documents import Document

chunk = Document(
    page_content=('결과로 생긴 중독 증상을 포함합니다. 그러나 음식물 섭취로 인한 증상, 세균성 음식물 중\n'
 '독과 상습적으로 흡입, 흡수 또는 섭취한 결과로 생긴 중독 증상은 포함되지 않습니다.【음식물】 반려동물이 일상 생활 중 보호자 또는 '
 '생산자의 의도와 상관 없이 섭취할 수 있는 모\n'
 '든 식이 원료와 가공품 및 부산물(뼈, 과일 씨 등 폐기 대상 물질)을 말하며, 사람 및 다른 동물의\n'
 '식이로 활용될 수 있는 모든 것을 포함합니다. 또한 음식물의 상태(부패, 감염 여부 등)와 상관없이'),
    metadata={'source_doc': {'total_pages': 35},
 'doc': {'doc_type': 'terms',
         'file_name': 'samsung_2_cat_anypet_3_20_21_37.pdf',
         'insurer_code': 'samsung',
         'product_code': '2',
         'product_name': '(일반)반려묘보험 애니펫',
         'total_pages': 1,
         'page': 1},
 'term_type': 'unknown',
 'clause': {'clause_type': 'other', 'risk_domains': ['other']},
 'indexing': {'chunk_id': 'chunk_000004',
              'chunk_char_len': 262,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'simple',
              'tag_method': 'rule',
              'tag_confidence': 0.25}},
)
