from langchain_core.documents import Document

chunk = Document(
    page_content=("style='font-size:16px'>가) 해당 관절의 운동범위 합계가 정상 운동범위의 1/2 이하로 제한된 경우</p><br><p "
 "id='115' data-category='list' style='font-size:16px'>나) 객관적 검사(스트레스 엑스선)상 "
 '10mm 이상의 동요관절(관절이 흔<br>들리거나 움직이는 것)이 있는 경우<br>다) 근전도 검사상 불완전한 손상(incomplete '
 'injury)소견이 있으면서<br>도수근력검사(MMT)에서 근력이 2등급(poor)인 경우<br>10) ‘관절 하나의 기능에 약간의'),
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
 'indexing': {'chunk_id': 'chunk_001604',
              'chunk_char_len': 299,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'normal',
              'tag_method': 'rule',
              'tag_confidence': 0.25}},
)
