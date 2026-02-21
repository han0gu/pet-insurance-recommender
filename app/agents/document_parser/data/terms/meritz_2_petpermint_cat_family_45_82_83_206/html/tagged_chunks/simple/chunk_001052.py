from langchain_core.documents import Document

chunk = Document(
    page_content=('. 단, 제거가 불가능한<br>경우에는 고정물 등이 있는 상태에서 장해를 평가한<br>다.<br>2) 관절을 사용하지 않아 발생한 '
 '일시적인 기능장해(예<br>를 들면 캐스트로 환부를 고정시켰기 때문에 치유 후<br>의 관절에 기능장해가 발생한 경우)는 장해로 '
 '평가<br>하지 않는다.<br>3) 손가락에는 첫째 손가락에 2개의 손가락관절이 있다.<br>그중 심장에서 가까운 쪽부터 중수지관절, '
 '지관절이<br>라 한다.<br>4) 다른 네 손가락에는 3개의 손가락관절이 있다'),
    metadata={'source_doc': {'total_pages': 160},
 'doc': {'doc_type': 'terms',
         'file_name': 'meritz_2_petpermint_cat_family_45_82_83_206.pdf',
         'insurer_code': 'meritz',
         'product_code': '2',
         'product_name': '무배당 펫퍼민트 Cat&Family보험 다이렉트2601',
         'total_pages': 1,
         'page': 1},
 'term_type': 'unknown',
 'clause': {'clause_type': 'other', 'risk_domains': ['digestive', 'joint']},
 'indexing': {'chunk_id': 'chunk_001052',
              'chunk_char_len': 264,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'simple',
              'tag_method': 'rule',
              'tag_confidence': 0.25}},
)
