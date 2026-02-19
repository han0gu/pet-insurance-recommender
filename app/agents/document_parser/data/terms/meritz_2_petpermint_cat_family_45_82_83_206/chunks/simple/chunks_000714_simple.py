from langchain_core.documents import Document

chunk = Document(
    page_content=('. 7) “손가락에 뚜렷한 장해를 남긴 때”라 함은 첫째 손가 락의 경우 중수지관절 또는 지관절의 굴신(굽히고 펴 기)운동영역이 정상 '
 '운동영역의 1/2 이하인 경우를 말 하며, 다른 네 손가락에서는 제1, 제2지관절의 굴신 (굽히고 펴기)운동영역을 합산하여 정상운동영역의 '
 '1/2 이하이거나 중수지관절의 굴신(굽히고 펴기)운동 영역이 정상운동영역의 1/2 이하인 경우를 말한다. 8) 한 손가락에 장해가 생기고 '
 '다른 손가락에 장해가 발 생한 경우, 지급률은 각각 적용하여 합산한다.'),
    metadata={'source_doc': {'total_pages': 160},
 'doc': {'doc_type': 'terms',
         'file_name': 'meritz_2_petpermint_cat_family_45_82_83_206.pdf',
         'insurer_code': 'meritz',
         'product_code': '2',
         'product_name': '무배당 펫퍼민트 Puppy&Family보험 다이렉트2601',
         'total_pages': 160,
         'page': 196},
 'term_type': 'special',
 'clause': {'clause_type': 'coverage', 'risk_domains': ['joint']},
 'indexing': {'chunk_id': 'chunk_000714',
              'chunk_char_len': 268,
              'embedding_model': 'solar-embedding-1-large',
              'tag_method': 'llm',
              'tag_confidence': 0.95}},
)
