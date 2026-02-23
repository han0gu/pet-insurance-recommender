from langchain_core.documents import Document

chunk = Document(
    page_content=('때’라 함은 아래의 경우 중<br>하나에 해당하는 때를 말한다.<br>가) 해당 관절의 운동범위 합계가 정상 운동범위의 3/4 이하로 '
 '제한된 경우<br>나) 근전도 검사상 불완전한 손상(incomplete injury)소견이 있으면서<br>도수근력검사(MMT)에서 근력이 '
 '3등급(fair)인 경우<br>11) ‘가관절주 \ue045 이 남아 뚜렷한 장해를 남긴 때’라 함은 상완골에 가관절이 남<br>은 경우 '
 '또는 요골과 척골의 2개 뼈 모두에 가관절이 남은 경우를 말한다.<br>주) 가관절이란, 충분한 경과 및 골이식술 등 골유합을 얻는데'),
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
 'indexing': {'chunk_id': 'chunk_001587',
              'chunk_char_len': 298,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'simple',
              'tag_method': 'rule',
              'tag_confidence': 0.25}},
)
