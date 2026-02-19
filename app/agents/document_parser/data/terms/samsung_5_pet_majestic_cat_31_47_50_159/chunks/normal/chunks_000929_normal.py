from langchain_core.documents import Document

chunk = Document(
    page_content=('. 3) "팔" 이라 함은 어깨관절(견관절)부터 손목관절(완관절)까지를 말한다. 4) "팔의 3대 관절" 이라 함은 어깨관절(견관절), '
 '팔꿈치관절(주관절), 손목관절 (완관절)을 말한다. 5) "한 팔의 손목 이상을 잃었을 때" 라 함은 손목관절(완관절)부터(손목관절 포 '
 '함) 심장에 가까운 쪽에서 절단된 때를 말하며, 팔꿈치관절(주관절) 상부에서 절단된 경우도 포함한다. 6) 팔의 관절기능장해 평가는 팔의 '
 '3대 관절의 관절운동범위 제한 등으로 평가한 다.'),
    metadata={'source_doc': {'total_pages': 129},
 'doc': {'doc_type': 'terms',
         'file_name': 'samsung_5_pet_majestic_cat_31_47_50_159.pdf',
         'insurer_code': 'samsung',
         'product_code': '5',
         'product_name': '(장기)무배당 삼성화재 펫보험 의기냥냥',
         'total_pages': 107,
         'page': 143},
 'term_type': 'special',
 'clause': {'clause_type': 'definition', 'risk_domains': ['joint']},
 'indexing': {'chunk_id': 'chunk_000929',
              'chunk_char_len': 256,
              'embedding_model': 'solar-embedding-1-large',
              'tag_method': 'llm',
              'tag_confidence': 0.95}},
)
