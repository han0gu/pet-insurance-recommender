from langchain_core.documents import Document

chunk = Document(
    page_content=('- 동영역을 합산하여 정상 운동영역의 1/2 이하이거나 중수지관절의 굴신(굽히\n'
 '- 고 펴기)운동영역이 정상 운동영역의 1/2 이하인 경우를 말한다.\n'
 '- 8) 한 손가락에 장해가 생기고 다른 손가락에 장해가 발생한 경우, 지급률은 각각\n'
 '- 적용하여 합산한다.\n'
 '- 9) 손가락의 관절기능장해 평가는 손가락 관절의 관절운동범위 제한 등으로 평가\n'
 '- 한다. 각 관절의 운동범위 측정은 장해평가시점의 「산업재해보상보험법 시행\n'
 '- 규칙」 제47조 제1항 및 제3항의 정상인의 신체 각 관절에 대한 평균 운동가'),
    metadata={'source_doc': {'total_pages': 129},
 'doc': {'doc_type': 'terms',
         'file_name': 'samsung_5_pet_majestic_cat_31_47_50_159.pdf',
         'insurer_code': 'samsung',
         'product_code': '5',
         'product_name': '(장기)무배당 삼성화재 펫보험 의기냥냥',
         'total_pages': 1,
         'page': 1},
 'term_type': 'unknown',
 'clause': {'clause_type': 'coverage', 'risk_domains': ['digestive', 'joint']},
 'indexing': {'chunk_id': 'chunk_000813',
              'chunk_char_len': 278,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'simple',
              'tag_method': 'rule',
              'tag_confidence': 0.8}},
)
