from langchain_core.documents import Document

chunk = Document(
    page_content=('. 7) "손가락에 뚜렷한 장해를 남긴 때" 라 함은 첫째 손가락의 경우 중수지관절 또 는 지관절의 굴신(굽히고 펴기)운동영역이 정상 '
 '운동영역의 1/2 이하인 경우 를 말하며, 다른 네 손가락에 있어서는 제1, 제2지관절의 굴신(굽히고 펴기)운 동영역을 합산하여 정상 '
 '운동영역의 1/2 이하이거나 중수지관절의 굴신(굽히 고 펴기)운동영역이 정상 운동영역의 1/2 이하인 경우를 말한다. 8) 한 손가락에 '
 '장해가 생기고 다른 손가락에 장해가 발생한 경우, 지급률은 각각 적용하여 합산한다'),
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
 'indexing': {'chunk_id': 'chunk_000954',
              'chunk_char_len': 271,
              'embedding_model': 'solar-embedding-1-large',
              'tag_method': 'llm',
              'tag_confidence': 0.95}},
)
