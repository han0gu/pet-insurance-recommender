from langchain_core.documents import Document

chunk = Document(
    page_content=('<용어풀이>\n'
 '[음식물]\n'
 '반려묘가 일상 생활 중 보호자 또는 생산자의 의도와 상관 없이 섭취할 수 있는 모든 식이 원료와 가공품 및 부산물(뼈, 과일 씨 등 폐기 '
 '대상물질)을 말하며, 사람 및 다른 동물의 식이로 활용될 수 있는 모든 것을 포함합니다. 또한 음식물의 상태(부패, 감염 여부 등)와 '
 '상관없이 모두 포함됩니다.'),
    metadata={'source_doc': {'total_pages': 129},
 'doc': {'doc_type': 'terms',
         'file_name': 'samsung_5_pet_majestic_cat_31_47_50_159.pdf',
         'insurer_code': 'samsung',
         'product_code': '5',
         'product_name': '(장기)무배당 삼성화재 펫보험 의기냥냥',
         'total_pages': 107,
         'page': 96},
 'term_type': 'special',
 'clause': {'clause_type': 'definition',
            'risk_domains': ['digestive', 'other']},
 'indexing': {'chunk_id': 'chunk_000526',
              'chunk_char_len': 179,
              'embedding_model': 'solar-embedding-1-large',
              'tag_method': 'llm',
              'tag_confidence': 0.95}},
)
