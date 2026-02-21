from langchain_core.documents import Document

chunk = Document(
    page_content=('- 중수지관절, 제1지관절(근위지관절) 및 제2지관절(원위지관절)이라 부른다.\n'
 '- 5) "손가락을 잃었을 때" 라 함은 첫째 손가락에서는 지관절부터 심장에서 가까\n'
 '- 운 쪽에서, 다른 네 손가락에서는 제1지관절(근위지관절)부터(제1지관절 포함)\n'
 '심장에서 가까운 쪽으로 손가락이 절단되었을 때를 말한다.- 6) "손가락뼈 일부를 잃었을 때" 라 함은 첫째 손가락의 지관절, 다른 네 '
 '손가락\n'
 '- 의 제1지관절(근위지관절)로부터 심장에서 먼 쪽으로 손가락 뼈의 일부가 절'),
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
 'indexing': {'chunk_id': 'chunk_000811',
              'chunk_char_len': 260,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'simple',
              'tag_method': 'rule',
              'tag_confidence': 0.25}},
)
