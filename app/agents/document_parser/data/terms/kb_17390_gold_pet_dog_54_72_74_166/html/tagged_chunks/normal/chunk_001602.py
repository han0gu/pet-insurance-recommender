from langchain_core.documents import Document

chunk = Document(
    page_content=('경우<br>8) ‘관절 하나의 기능에 심한 장해를 남긴 때’라 함은 아래의 경우 중 하<br>나에 해당하는 때를 '
 "말한다.</p><br><p id='112' data-category='list' style='font-size:16px'>가) 해당 "
 '관절의 운동범위 합계가 정상 운동범위의 1/4 이하로 제한된 경우<br>나) 인공관절이나 인공골두를 삽입한 경우<br>다) 객관적 '
 '검사(스트레스 엑스선)상 15mm 이상의 동요관절(관절이 흔<br>들리거나 움직이는 것)이 있는 경우<br>라) 근전도 검사상 '
 '완전손상(complete'),
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
 'indexing': {'chunk_id': 'chunk_001602',
              'chunk_char_len': 294,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'normal',
              'tag_method': 'rule',
              'tag_confidence': 0.25}},
)
