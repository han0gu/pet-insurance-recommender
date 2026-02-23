from langchain_core.documents import Document

chunk = Document(
    page_content=('일한 부위로 본다.# 2) "골반뼈의 뚜렷한 기형" 이라 함은 아래의 경우 중 하나에 해당하는 때를 말한\n'
 '다.- 가) 천장관절 또는 치골문합부가 분리된 상태로 치유되었거나 좌골이 2.5cm이\n'
 '- 상 분리된 부정유합 상태\n'
 '- 나) 육안으로 변형(결손을 포함)을 명백하게 알 수 있을 정도로 방사선 검사로\n'
 '- 측정한 각(角) 변형이 20° 이상인 경우\n'
 '- 다) 미골의 기형은 골절이나 탈구로 방사선 검사로 측정한 각(角) 변형이 70°\n'
 '- 이상 남은 상태'),
    metadata={'source_doc': {'total_pages': 129},
 'doc': {'doc_type': 'terms',
         'file_name': 'samsung_5_pet_majestic_cat_31_47_50_159.pdf',
         'insurer_code': 'samsung',
         'product_code': '5',
         'product_name': '(장기)무배당 삼성화재 펫보험 의기냥냥',
         'total_pages': 1,
         'page': 1},
 'term_type': 'unknown',
 'clause': {'clause_type': 'other', 'risk_domains': ['digestive', 'joint']},
 'indexing': {'chunk_id': 'chunk_000782',
              'chunk_char_len': 252,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'simple',
              'tag_method': 'rule',
              'tag_confidence': 0.25}},
)
