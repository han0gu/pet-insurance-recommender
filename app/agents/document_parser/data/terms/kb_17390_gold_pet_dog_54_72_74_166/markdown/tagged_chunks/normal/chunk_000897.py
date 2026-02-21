from langchain_core.documents import Document

chunk = Document(
    page_content=('- 도수근력검사(MMT)에서 근력이 2등급(poor)인 경우\n'
 '- 10) ‘관절 하나의 기능에 약간의 장해를 남긴 때’라 함은 아래의 경우 중\n'
 '- 하나에 해당하는 때를 말한다.\n'
 '- 가) 해당 관절의 운동범위 합계가 정상 운동범위의 3/4 이하로 제한된 경우\n'
 '- 나) 근전도 검사상 불완전한 손상(incomplete injury)소견이 있으면서\n'
 '- 도수근력검사(MMT)에서 근력이 3등급(fair)인 경우\n'
 '- 11) ‘가관절주 \ue045 이 남아 뚜렷한 장해를 남긴 때’라 함은 상완골에 가관절이 남'),
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
 'indexing': {'chunk_id': 'chunk_000897',
              'chunk_char_len': 273,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'normal',
              'tag_method': 'rule',
              'tag_confidence': 0.25}},
)
