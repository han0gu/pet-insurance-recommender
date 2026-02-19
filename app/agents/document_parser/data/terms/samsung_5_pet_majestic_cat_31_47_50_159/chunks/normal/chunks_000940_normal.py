from langchain_core.documents import Document

chunk = Document(
    page_content=('. 3) "다리" 라 함은 엉덩이관절(고관절)부터 발목관절(족관절)까지를 말한다. 4) "다리의 3대 관절" 이라 함은 '
 '엉덩이관절(고관절), 무릎관절(슬관절), 발목관절 (족관절)을 말한다. 5) "한 다리의 발목 이상을 잃었을 때" 라 함은 '
 '발목관절(족관절)부터(발목관절 포함) 심장에 가까운 쪽에서 절단된 때를 말하며, 무릎관절(슬관절)의 상부에 서 절단된 경우도 포함한다. '
 '6) 다리의 관절기능장해 평가는 다리의 3대 관절의 관절운동범위 제한 및 무릎관 절(슬관절)의 동요성 등으로 평가한다.'),
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
 'indexing': {'chunk_id': 'chunk_000940',
              'chunk_char_len': 278,
              'embedding_model': 'solar-embedding-1-large',
              'tag_method': 'llm',
              'tag_confidence': 0.95}},
)
