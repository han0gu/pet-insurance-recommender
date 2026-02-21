from langchain_core.documents import Document

chunk = Document(
    page_content=('절(슬관절)의 동요성 등으로 평가한다.- \n'
 '- 가) 각 관절의 운동범위 측정은 장해평가시점의 「산업재해보상보험법 시행규\n'
 '- 칙」 제47조 제1항 및 제3항의 정상인의 신체 각 관절에 대한 평균 운동가\n'
 '- 능영역을 기준으로 정상각도 및 측정방법 등을 따른다.\n'
 '- 나) 관절기능장해가 신경손상으로 인한 경우에는 운동범위 측정이 아닌 근력\n'
 '- 및 근전도 검사를 기준으로 평가한다.\n'
 '7) "관절 하나의 기능을 완전히 잃었을 때" 라 함은 아래의 경우 중 하나에 해당'),
    metadata={'source_doc': {'total_pages': 129},
 'doc': {'doc_type': 'terms',
         'file_name': 'samsung_5_pet_majestic_cat_31_47_50_159.pdf',
         'insurer_code': 'samsung',
         'product_code': '5',
         'product_name': '(장기)무배당 삼성화재 펫보험 의기냥냥',
         'total_pages': 1,
         'page': 1},
 'term_type': 'unknown',
 'clause': {'clause_type': 'coverage',
            'risk_domains': ['digestive', 'head', 'joint']},
 'indexing': {'chunk_id': 'chunk_000801',
              'chunk_char_len': 256,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'simple',
              'tag_method': 'rule',
              'tag_confidence': 0.8}},
)
