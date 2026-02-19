from langchain_core.documents import Document

chunk = Document(
    page_content=('1) 골절부에 금속내고정물 등을 사용하였기 때문에 그것이 기능장해의 원인이 되 는 때에는 그 내고정물 등이 제거된 후에 장해를 평가한다. '
 '단, 제거가 불가능 한 경우에는 고정물 등이 있는 상태에서 장해를 평가한다. 2) 관절을 사용하지 않아 발생한 일시적인 기능장해(예를 '
 '들면 캐스트로 환부를 고 정시켰기 때문에 치유 후의 관절에 기능장해가 발생한 경우)는 장해로 평가하 지 않는다. 3) 손가락에는 첫째 '
 '손가락에 2개의 손가락관절이 있다. 그 중 심장에서 가까운 쪽 부터 중수지관절, 지관절이라 한다'),
    metadata={'source_doc': {'total_pages': 129},
 'doc': {'doc_type': 'terms',
         'file_name': 'samsung_5_pet_majestic_cat_31_47_50_159.pdf',
         'insurer_code': 'samsung',
         'product_code': '5',
         'product_name': '(장기)무배당 삼성화재 펫보험 의기냥냥',
         'total_pages': 107,
         'page': 145},
 'term_type': 'special',
 'clause': {'clause_type': 'exclusion', 'risk_domains': ['joint', 'other']},
 'indexing': {'chunk_id': 'chunk_000950',
              'chunk_char_len': 280,
              'embedding_model': 'solar-embedding-1-large',
              'tag_method': 'llm',
              'tag_confidence': 0.95}},
)
