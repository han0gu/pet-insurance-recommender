from langchain_core.documents import Document

chunk = Document(
    page_content=('. 9) 손가락의 관절기능장해 평가는 손가락 관절의 관절운동범위 제한 등으로 평가 한다. 각 관절의 운동범위 측정은 장해평가시점의 '
 '「산업재해보상보험법 시행 규칙」 제47조 제1항 및 제3항의 정상인의 신체 각 관절에 대한 평균 운동가 능영역을 기준으로 정상각도 및 '
 '측정방법 등을 따른다.'),
    metadata={'source_doc': {'total_pages': 129},
 'doc': {'doc_type': 'terms',
         'file_name': 'samsung_5_pet_majestic_cat_31_47_50_159.pdf',
         'insurer_code': 'samsung',
         'product_code': '5',
         'product_name': '(장기)무배당 삼성화재 펫보험 의기냥냥',
         'total_pages': 107,
         'page': 145},
 'term_type': 'special',
 'clause': {'clause_type': 'coverage', 'risk_domains': ['digestive', 'joint']},
 'indexing': {'chunk_id': 'chunk_000955',
              'chunk_char_len': 160,
              'embedding_model': 'solar-embedding-1-large',
              'tag_method': 'rule',
              'tag_confidence': 0.8}},
)
