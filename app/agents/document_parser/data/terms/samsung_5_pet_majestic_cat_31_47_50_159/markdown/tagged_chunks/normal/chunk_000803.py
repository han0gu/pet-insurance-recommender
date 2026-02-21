from langchain_core.documents import Document

chunk = Document(
    page_content=('- 나) 인공관절이나 인공골두를 삽입한 경우\n'
 '- 다) 객관적 검사(스트레스 엑스선)상 15mm 이상의 동요관절(관절이 흔들리거\n'
 '- 나 움직이는 것)이 있는 경우\n'
 '- 라) 근전도 검사상 완전손상(complete injury) 소견이 있으면서 도수근력검사\n'
 '- (MMT)에서 근력이 "1등급(trace)" 인 경우\n'
 '- 9) "관절 하나의 기능에 뚜렷한 장해를 남긴 때" 라 함은 아래의 경우 중 하나에\n'
 '- 해당하는 때를 말한다.\n'
 '- 가) 해당 관절의 운동범위 합계가 정상 운동범위의 1/2 이하로 제한된 경우'),
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
 'indexing': {'chunk_id': 'chunk_000803',
              'chunk_char_len': 281,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'normal',
              'tag_method': 'rule',
              'tag_confidence': 0.25}},
)
