from langchain_core.documents import Document

chunk = Document(
    page_content=('- 한 경우에는 고정물 등이 있는 상태에서 장해를 평가한다.\n'
 '- 2) 관절을 사용하지 않아 발생한 일시적인 기능장해(예를 들면 캐스트로 환부를 고\n'
 '- 정시켰기 때문에 치유 후의 관절에 기능장해가 발생한 경우)는 장해로 평가하\n'
 '- 지 않는다.\n'
 '- 3) 손가락에는 첫째 손가락에 2개의 손가락관절이 있다. 그 중 심장에서 가까운 쪽\n'
 '- 부터 중수지관절, 지관절이라 한다.\n'
 '- 4) 다른 네 손가락에는 3개의 손가락관절이 있다. 그 중 심장에서 가까운 쪽부터\n'
 '- 중수지관절, 제1지관절(근위지관절) 및 제2지관절(원위지관절)이라 부른다.'),
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
 'indexing': {'chunk_id': 'chunk_000810',
              'chunk_char_len': 297,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'simple',
              'tag_method': 'rule',
              'tag_confidence': 0.25}},
)
