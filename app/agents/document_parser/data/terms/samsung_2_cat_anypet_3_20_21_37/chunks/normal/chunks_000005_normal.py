from langchain_core.documents import Document

chunk = Document(
    page_content=('【음식물】 반려동물이 일상 생활 중 보호자 또는 생산자의 의도와 상관 없이 섭취할 수 있는 모 든 식이 원료와 가공품 및 부산물(뼈, '
 '과일 씨 등 폐기 대상 물질)을 말하며, 사람 및 다른 동물의 식이로 활용될 수 있는 모든 것을 포함합니다. 또한 음식물의 상태(부패, '
 '감염 여부 등)와 상관없이 모두 포함됩니다.\n'
 '나. 질병: 상해를 제외한 상병을 포함합니다. 단, 약관에서 명기하는 보상하지 않는 질병은 제외 합니다.\n'
 '3. 보상 관련 용어'),
    metadata={'source_doc': {'total_pages': 35},
 'doc': {'doc_type': 'terms',
         'file_name': 'samsung_2_cat_anypet_3_20_21_37.pdf',
         'insurer_code': 'samsung',
         'product_code': '2',
         'product_name': '(일반)반려묘보험 애니펫',
         'total_pages': 35,
         'page': 4},
 'term_type': 'basic',
 'clause': {'clause_type': 'exclusion', 'risk_domains': ['other']},
 'indexing': {'chunk_id': 'chunk_000005',
              'chunk_char_len': 246,
              'embedding_model': 'solar-embedding-1-large',
              'tag_method': 'rule',
              'tag_confidence': 0.85}},
)
