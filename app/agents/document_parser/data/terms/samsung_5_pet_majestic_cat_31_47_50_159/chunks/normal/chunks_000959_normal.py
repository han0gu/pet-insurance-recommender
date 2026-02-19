from langchain_core.documents import Document

chunk = Document(
    page_content=('. 3) "발가락을 잃었을 때" 라 함은 첫째 발가락에서는 지관절부터 심장에 가까운 쪽을, 나머지 네 발가락에서는 '
 '제1지관절(근위지관절)부터(제1지관절 포함) 심 장에서 가까운 쪽을 잃었을 때를 말한다. 4) 리스프랑 관절 이상에서 잃은 때라 함은 '
 '족근-중족골간 관절 이상에서 절단된 경우를 말한다'),
    metadata={'source_doc': {'total_pages': 129},
 'doc': {'doc_type': 'terms',
         'file_name': 'samsung_5_pet_majestic_cat_31_47_50_159.pdf',
         'insurer_code': 'samsung',
         'product_code': '5',
         'product_name': '(장기)무배당 삼성화재 펫보험 의기냥냥',
         'total_pages': 107,
         'page': 146},
 'term_type': 'special',
 'clause': {'clause_type': 'definition', 'risk_domains': ['joint']},
 'indexing': {'chunk_id': 'chunk_000959',
              'chunk_char_len': 166,
              'embedding_model': 'solar-embedding-1-large',
              'tag_method': 'llm',
              'tag_confidence': 0.95}},
)
