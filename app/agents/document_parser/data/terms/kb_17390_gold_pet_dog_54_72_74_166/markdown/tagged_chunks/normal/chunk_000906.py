from langchain_core.documents import Document

chunk = Document(
    page_content=('- 들리거나 움직이는 것)이 있는 경우\n'
 '- 라) 근전도 검사상 완전손상(complete injury) 소견이 있으면서 도수근\n'
 '- 력검사(MMT)에서 근력이 ‘1등급(trace)’인 경우\n'
 '9) ‘관절 하나의 기능에 뚜렷한 장해를 남긴 때’라 함은 아래의 경우 중\n'
 '하나에 해당하는 때를 말한다.가) 해당 관절의 운동범위 합계가 정상 운동범위의 1/2 이하로 제한된 경우- 나) 객관적 검사(스트레스 '
 '엑스선)상 10mm 이상의 동요관절(관절이 흔\n'
 '- 들리거나 움직이는 것)이 있는 경우'),
    metadata={'source_doc': {'total_pages': 113},
 'doc': {'doc_type': 'terms',
         'file_name': 'kb_17390_gold_pet_dog_54_72_74_166.pdf',
         'insurer_code': 'kb',
         'product_code': '17390',
         'product_name': '[일반보험] KB반려행복펫보험',
         'total_pages': 1,
         'page': 1},
 'term_type': 'unknown',
 'clause': {'clause_type': 'other', 'risk_domains': ['digestive', 'joint']},
 'indexing': {'chunk_id': 'chunk_000906',
              'chunk_char_len': 268,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'normal',
              'tag_method': 'rule',
              'tag_confidence': 0.25}},
)
