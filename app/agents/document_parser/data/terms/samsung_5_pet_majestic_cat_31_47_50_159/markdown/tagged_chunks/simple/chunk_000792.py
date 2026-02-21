from langchain_core.documents import Document

chunk = Document(
    page_content=('- (MMT)에서 근력이 "0등급(zero)" 인 경우\n'
 '# 8) "관절 하나의 기능에 심한 장해를 남긴 때" 라 함은 아래의 경우 중 하나에 해\n'
 '당하는 경우를 말한다.가) 해당 관절의 운동범위 합계가 정상 운동범위의 1/4 이하로 제한된 경우\n'
 '나) 인공관절이나 인공골두를 삽입한 경우- \n'
 '다) 근전도 검사상 완전손상(complete injury) 소견이 있으면서 도수근력검사\n'
 '(MMT)에서 근력이 "1등급(trace)" 인 경우- \n'
 '9) "관절 하나의 기능에 뚜렷한 장해를 남긴 때" 라 함은 아래의 경우 중 하나에'),
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
 'indexing': {'chunk_id': 'chunk_000792',
              'chunk_char_len': 288,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'simple',
              'tag_method': 'rule',
              'tag_confidence': 0.25}},
)
