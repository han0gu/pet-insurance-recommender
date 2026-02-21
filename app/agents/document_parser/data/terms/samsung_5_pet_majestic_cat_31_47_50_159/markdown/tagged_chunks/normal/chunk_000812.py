from langchain_core.documents import Document

chunk = Document(
    page_content=('- 의 제1지관절(근위지관절)로부터 심장에서 먼 쪽으로 손가락 뼈의 일부가 절\n'
 '- 단된 경우를 말하며, 뼈 단면이 불규칙해진 상태나 손가락 길이의 단축 없이\n'
 '- 골편만 떨어진 상태는 해당하지 않는다.\n'
 '- 7) "손가락에 뚜렷한 장해를 남긴 때" 라 함은 첫째 손가락의 경우 중수지관절 또\n'
 '- 는 지관절의 굴신(굽히고 펴기)운동영역이 정상 운동영역의 1/2 이하인 경우\n'
 '- 를 말하며, 다른 네 손가락에 있어서는 제1, 제2지관절의 굴신(굽히고 펴기)운\n'
 '- 동영역을 합산하여 정상 운동영역의 1/2 이하이거나 중수지관절의 굴신(굽히'),
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
 'indexing': {'chunk_id': 'chunk_000812',
              'chunk_char_len': 296,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'normal',
              'tag_method': 'rule',
              'tag_confidence': 0.25}},
)
