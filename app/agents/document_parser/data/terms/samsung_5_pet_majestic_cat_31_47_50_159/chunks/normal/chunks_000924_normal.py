from langchain_core.documents import Document

chunk = Document(
    page_content=('좌골 포함), 빗장뼈(쇄골), 가슴뼈(흉골), 갈비뼈(늑골)를 말하며 이를 모두 동 일한 부위로 본다.\n'
 '2) "골반뼈의 뚜렷한 기형" 이라 함은 아래의 경우 중 하나에 해당하는 때를 말한 다.\n'
 '가) 천장관절 또는 치골문합부가 분리된 상태로 치유되었거나 좌골이 2.5cm이 상 분리된 부정유합 상태 나) 육안으로 변형(결손을 '
 '포함)을 명백하게 알 수 있을 정도로 방사선 검사로 측정한 각(角) 변형이 20° 이상인 경우 다) 미골의 기형은 골절이나 탈구로 방사선 '
 '검사로 측정한 각(角) 변형이 70° 이상 남은 상태'),
    metadata={'source_doc': {'total_pages': 129},
 'doc': {'doc_type': 'terms',
         'file_name': 'samsung_5_pet_majestic_cat_31_47_50_159.pdf',
         'insurer_code': 'samsung',
         'product_code': '5',
         'product_name': '(장기)무배당 삼성화재 펫보험 의기냥냥',
         'total_pages': 107,
         'page': 142},
 'term_type': 'special',
 'clause': {'clause_type': 'definition', 'risk_domains': ['joint', 'other']},
 'indexing': {'chunk_id': 'chunk_000924',
              'chunk_char_len': 287,
              'embedding_model': 'solar-embedding-1-large',
              'tag_method': 'llm',
              'tag_confidence': 0.95}},
)
