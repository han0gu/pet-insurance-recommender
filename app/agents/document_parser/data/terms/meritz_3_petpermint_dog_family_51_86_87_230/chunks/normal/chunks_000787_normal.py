from langchain_core.documents import Document

chunk = Document(
    page_content=('1) 골절부에 금속내고정물 등을 사용하였기 때문에 그것 이 기능장해의 원인이 되는 때에는 그 내고정물 등이 제거된 후에 장해를 평가한다. '
 '단, 제거가 불가능한 경우에는 고정물 등이 있는 상태에서 장해를 평가한 다. 2) 관절을 사용하지 않아 발생한 일시적인 기능장해(예 를 '
 '들면 캐스트로 환부를 고정시켰기 때문에 치유 후 의 관절에 기능장해가 발생한 경우)는 장해로 평가 하지 않는다. 3) 손가락에는 첫째 '
 '손가락에 2개의 손가락관절이 있다. 그중 심장에서 가까운 쪽부터 중수지관절, 지관절이 라 한다'),
    metadata={'source_doc': {'total_pages': 180},
 'doc': {'doc_type': 'terms',
         'file_name': 'meritz_3_petpermint_dog_family_51_86_87_230.pdf',
         'insurer_code': 'meritz',
         'product_code': '3',
         'product_name': '무배당 펫퍼민트 Puppy&Family보험 다이렉트2601',
         'total_pages': 180,
         'page': 221},
 'term_type': 'special',
 'clause': {'clause_type': 'exclusion', 'risk_domains': ['joint', 'other']},
 'indexing': {'chunk_id': 'chunk_000787',
              'chunk_char_len': 280,
              'embedding_model': 'solar-embedding-1-large',
              'tag_method': 'llm',
              'tag_confidence': 0.95}},
)
