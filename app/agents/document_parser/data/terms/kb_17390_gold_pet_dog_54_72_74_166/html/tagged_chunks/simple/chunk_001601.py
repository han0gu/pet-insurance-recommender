from langchain_core.documents import Document

chunk = Document(
    page_content=("경우 중 하나에</p><br><p id='110' data-category='list' style='font-size:16px'>해당하는 "
 '때를 말한다.<br>가) 완전 강직(관절굳음)<br>나) 근전도 검사상 완전손상(complete injury) 소견이 있으면서 '
 "도수근</p><br><p id='111' data-category='paragraph' "
 "style='font-size:16px'>력검사(MMT)에서 근력이 ‘0등급(zero)’인 경우<br>8) ‘관절 하나의 기능에 심한 "
 '장해를 남긴 때’라 함은 아래의 경우 중'),
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
 'indexing': {'chunk_id': 'chunk_001601',
              'chunk_char_len': 294,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'simple',
              'tag_method': 'rule',
              'tag_confidence': 0.25}},
)
