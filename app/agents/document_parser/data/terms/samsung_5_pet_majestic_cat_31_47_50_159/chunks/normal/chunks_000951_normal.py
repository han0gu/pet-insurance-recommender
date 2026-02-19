from langchain_core.documents import Document

chunk = Document(
    page_content=('. 그 중 심장에서 가까운 쪽 부터 중수지관절, 지관절이라 한다. 4) 다른 네 손가락에는 3개의 손가락관절이 있다. 그 중 심장에서 '
 '가까운 쪽부터 중수지관절, 제1지관절(근위지관절) 및 제2지관절(원위지관절)이라 부른다. 5) "손가락을 잃었을 때" 라 함은 첫째 '
 '손가락에서는 지관절부터 심장에서 가까 운 쪽에서, 다른 네 손가락에서는 제1지관절(근위지관절)부터(제1지관절 포함)'),
    metadata={'source_doc': {'total_pages': 129},
 'doc': {'doc_type': 'terms',
         'file_name': 'samsung_5_pet_majestic_cat_31_47_50_159.pdf',
         'insurer_code': 'samsung',
         'product_code': '5',
         'product_name': '(장기)무배당 삼성화재 펫보험 의기냥냥',
         'total_pages': 107,
         'page': 145},
 'term_type': 'special',
 'clause': {'clause_type': 'definition', 'risk_domains': ['joint']},
 'indexing': {'chunk_id': 'chunk_000951',
              'chunk_char_len': 213,
              'embedding_model': 'solar-embedding-1-large',
              'tag_method': 'llm',
              'tag_confidence': 0.95}},
)
