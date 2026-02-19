from langchain_core.documents import Document

chunk = Document(
    page_content=('가) 각 관절의 운동범위 측정은 장해평가시점의 「산업재해보상보험법 시행규 칙」 제47조 제1항 및 제3항의 정상인의 신체 각 관절에 대한 '
 '평균 운동가 능영역을 기준으로 정상각도 및 측정방법 등을 따른다. 나) 관절기능장해를 표시할 경우 장해부위의 장해각도와 정상부위의 '
 '측정치를 동시에 판단하여 장해상태를 명확히 한다. 단, 관절기능장해가 신경손상으 로 인한 경우에는 운동범위 측정이 아닌 근력 및 근전도 '
 '검사를 기준으로 평가한다.\n'
 '7) "관절 하나의 기능을 완전히 잃었을 때" 라 함은 아래의 경우 중 하나에 해당 하는 경우를 말한다.'),
    metadata={'source_doc': {'total_pages': 129},
 'doc': {'doc_type': 'terms',
         'file_name': 'samsung_5_pet_majestic_cat_31_47_50_159.pdf',
         'insurer_code': 'samsung',
         'product_code': '5',
         'product_name': '(장기)무배당 삼성화재 펫보험 의기냥냥',
         'total_pages': 107,
         'page': 143},
 'term_type': 'special',
 'clause': {'clause_type': 'definition', 'risk_domains': ['joint']},
 'indexing': {'chunk_id': 'chunk_000930',
              'chunk_char_len': 299,
              'embedding_model': 'solar-embedding-1-large',
              'tag_method': 'llm',
              'tag_confidence': 0.95}},
)
